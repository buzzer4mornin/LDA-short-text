import numpy as np
import sys

sys.path.insert(0, '../common')
import utilities


class INFERENCE_fw:
    """
    Compute perplexity, employing Frank-Wolfe algorithm.
    """

    def __init__(self, beta, max_iter):
        """
        Arguments:
            beta: Topics of the learned model.
            max_infer: Number of iterations of FW algorithm.
        """
        self.beta = np.copy(beta) + 1e-10
        self.num_topics = beta.shape[0]
        self.num_terms = beta.shape[1]
        self.INF_MAX_ITER = max_iter

        # Normalize beta
        beta_norm = self.beta.sum(axis=1)
        self.beta /= beta_norm[:, np.newaxis]
        self.logbeta = np.log(self.beta)

        # Generate values used for initilization of topic mixture of each document
        self.theta_init = [1e-10] * self.num_topics
        self.theta_vert = 1. - 1e-10 * (self.num_topics - 1)

    def e_step(self, batch_size, wordids, wordcts):
        """
        Infer topic mixtures (theta) for all document in 'w_obs' part.
        """
        # Declare theta of minibatch
        theta = np.zeros((batch_size, self.num_topics))
        # Do inference for each document
        for d in range(batch_size):
            thetad = self.infer_doc(wordids[d], wordcts[d])
            theta[d, :] = thetad
        return (theta)

    def infer_doc(self, ids, cts):
        """
        Infer topic mixture (theta) for each document in 'w_obs' part.
        """
        # Locate cache memory
        beta = self.beta[:, ids]
        logbeta = self.logbeta[:, ids]
        # Initialize theta to be a vertex of unit simplex
        # with the largest value of the objective function
        theta = np.array(self.theta_init)
        f = np.dot(logbeta, cts)
        index = np.argmax(f)
        theta[index] = self.theta_vert
        # x = sum_(k=2)^K theta_k * beta_{kj}
        x = np.copy(beta[index, :])
        # Loop
        for l in range(0, self.INF_MAX_ITER):
            # Select a vertex with the largest value of
            # derivative of the objective function
            df = np.dot(beta, cts / x)
            index = np.argmax(df)
            beta_x = beta[index, :] - x
            alpha = 2. / (l + 3)
            # Update theta
            theta *= 1 - alpha
            theta[index] += alpha
            # Update x
            x += alpha * (beta_x)
        return (theta)


beta = np.load("../output-data/beta.npy")
docs_file = f"../input-data/docs.txt"
setting_file = f"../input-data/settings.txt"
ddict = utilities.read_setting(setting_file)
wordids, wordcts = utilities.read_data(docs_file)
ml_ope = INFERENCE_fw(beta, ddict["iter_infer"])
theta = ml_ope.e_step(ddict["num_docs"], wordids, wordcts)
np.save("../output-data/theta.npy", theta)
