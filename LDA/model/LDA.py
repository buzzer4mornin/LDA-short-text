# -*- coding: utf-8 -*-
import numpy as np


class MyLDA:
    def __init__(self, num_terms, num_topics, alpha, tau0, kappa, BOPE, iter_infer):
        """ Click to read more

        Arguments:
            num_terms: Number of unique terms in the corpus (length of the vocabulary).
            num_topics: Number of topics shared by the whole corpus.
            alpha: Hyperparameter for prior on topic mixture theta.
            iter_infer: Number of iterations for BOPE algorithm
          """
        self.num_terms = num_terms
        self.num_topics = num_topics
        self.alpha = alpha
        self.tau0 = tau0
        self.kappa = kappa
        self.updatect = 1
        self.is_BOPE = True if BOPE == "true" else False
        self.INF_MAX_ITER = iter_infer

        # Initialize beta (topics)
        self.beta = np.random.rand(self.num_topics, self.num_terms) + 1e-10
        beta_norm = self.beta.sum(axis=1)
        self.beta /= beta_norm[:, np.newaxis]

    def run_EM(self, batch_size, wordids, wordcts):
        """
        First does an E step on the mini-batch given in wordids and
        wordcts, then uses the result of that E step to update the
        topics in M step.

        Arguments:
        batch_size: Number of documents of the mini-batch.
        wordids: A list whose each element is an array (terms), corresponding to a document.
                 Each element of the array is index of a unique term, which appears in the document,
                 in the vocabulary.
        wordcts: A list whose each element is an array (frequency), corresponding to a document.
                 Each element of the array says how many time the corresponding term in wordids appears
                 in the document.
        Returns time the E and M steps have taken and the list of topic mixtures of all documents in the mini-batch.
        """
        # E step
        theta = self.e_step(batch_size, wordids, wordcts)

        # M step
        self.m_step(batch_size, wordids, wordcts, theta)

        return theta

    def e_step(self, batch_size, wordids, wordcts):
        """
        Does e step

        Returns topic mixtures theta.
        """
        # Declare theta of minibatch
        theta = np.zeros((batch_size, self.num_topics))
        # Inference
        for d in range(batch_size):
            thetad = self.infer_doc(wordids[d], wordcts[d])
            theta[d, :] = thetad
        return theta

    def infer_doc(self, ids, cts):
        """
        Does inference for a document using OPE or BOPE.

        Arguments:
        ids: an element of wordids, corresponding to a document.
        cts: an element of wordcts, corresponding to a document.

        Returns inferred theta.
        """
        # locate cache memory
        beta = self.beta[:, ids]

        # Initialize theta randomly
        theta = np.random.rand(self.num_topics) + 1.
        theta /= sum(theta)

        # x = sum_(k=2)^K theta_k * beta_{kj}
        x = np.dot(theta, beta)

        if self.is_BOPE:
            # Parameter of Bernoulli distribution
            # Likelihood vs Prior
            p = 0.9

            # Number of times likelihood and prior are chosen
            T_lower = [1, 0]
            T_upper = [0, 1]

            for t in range(1, self.INF_MAX_ITER):
                # ======== Lower ==========
                if np.random.rand() < p:
                    T_lower[0] += 1
                else:
                    T_lower[1] += 1

                G_1 = np.dot(beta, cts / x) / p
                G_2 = (self.alpha - 1) / theta / (1 - p)

                ft_lower = T_lower[0] * G_1 + T_lower[1] * G_2
                index_lower = np.argmax(ft_lower)
                alpha = 1.0 / (t + 1)
                theta_lower = np.copy(theta)
                theta_lower *= 1 - alpha
                theta_lower[index_lower] += alpha

                # ======== Upper ==========
                if np.random.rand() < p:
                    T_upper[0] += 1
                else:
                    T_upper[1] += 1

                ft_upper = T_upper[0] * G_1 + T_upper[1] * G_2
                index_upper = np.argmax(ft_upper)
                alpha = 1.0 / (t + 1)
                theta_upper = np.copy(theta)
                theta_upper *= 1 - alpha
                theta_upper[index_upper] += alpha
                # print(theta_upper - theta_lower)

                # ======== Decision ========
                x_l = np.dot(cts, np.log(np.dot(theta_lower, beta))) + (self.alpha - 1) * np.log(theta_lower)
                x_u = np.dot(cts, np.log(np.dot(theta_upper, beta))) + (self.alpha - 1) * np.log(theta_upper)

                compare = np.array([x_l[0], x_u[0]])
                best = np.argmax(compare)

                # ======== Update ========
                if best == 0:
                    theta = np.copy(theta_lower)
                    x = x + alpha * (beta[index_lower, :] - x)
                else:
                    theta = np.copy(theta_upper)
                    x = x + alpha * (beta[index_upper, :] - x)
        else:
            T = [1, 0]
            for t in range(1, self.INF_MAX_ITER):
                # Pick fi uniformly
                T[np.random.randint(2)] += 1
                # Select a vertex with the largest value of
                # derivative of the function F
                df = T[0] * np.dot(beta, cts / x) + T[1] * (self.alpha - 1) / theta
                index = np.argmax(df)
                alpha = 1.0 / (t + 1)
                # Update theta
                theta *= 1 - alpha
                theta[index] += alpha
                # Update x
                x = x + alpha * (beta[index, :] - x)
        return theta

    def m_step(self, batch_size, wordids, wordcts, theta):
        """
        Does m step: update global variables beta.
        """
        # Compute intermediate beta which is denoted as "unit beta"
        beta = np.zeros((self.num_topics, self.num_terms), dtype=float)
        for d in range(batch_size):
            beta[:, wordids[d]] += np.outer(theta[d], wordcts[d])
        # Check zeros index
        beta_sum = beta.sum(axis=0)
        ids = np.where(beta_sum != 0)[0]
        unit_beta = beta[:, ids]
        # Normalize the intermediate beta
        unit_beta_norm = unit_beta.sum(axis=1)
        unit_beta /= unit_beta_norm[:, np.newaxis]
        # Update beta
        rhot = pow(self.tau0 + self.updatect, -self.kappa)
        self.rhot = rhot
        self.beta *= (1 - rhot)
        self.beta[:, ids] += unit_beta * rhot
        self.updatect += 1