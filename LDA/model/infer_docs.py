import numpy as np
import sys
sys.path.insert(0, '../common')
import utilities

class INFERENCE:
    def __init__(self, num_terms, num_topics, alpha, beta, iter_infer):
        self.num_topics = num_topics
        self.num_terms = num_terms
        self.alpha = alpha
        self.INF_MAX_ITER = iter_infer
        self.beta = beta

    def static_online(self, batch_size, wordids, wordcts):
        theta = self.e_step(batch_size, wordids, wordcts)
        return theta

    def e_step(self, batch_size, wordids, wordcts):
        # Declare theta of minibatch
        theta = np.zeros((batch_size, self.num_topics))
        # Inference
        for d in range(batch_size):
            thetad = self.infer_doc(wordids[d], wordcts[d])
            theta[d, :] = thetad
        return theta

    def infer_doc(self, ids, cts):
        # locate cache memory
        beta = self.beta[:, ids]
        # Initialize theta randomly
        theta = np.random.rand(self.num_topics) + 1.
        theta /= sum(theta)
        # x = sum_(k=2)^K theta_k * beta_{kj}
        x = np.dot(theta, beta)
        # Loop
        T = [1, 0]
        for l in range(1, self.INF_MAX_ITER):
            # Pick fi uniformly
            T[np.random.randint(2)] += 1
            # Select a vertex with the largest value of
            # derivative of the function F
            df = T[0] * np.dot(beta, cts / x) + T[1] * (self.alpha - 1) / theta
            index = np.argmax(df)
            alpha = 1.0 / (l + 1)
            # Update theta
            theta *= 1 - alpha
            theta[index] += alpha
            # Update x
            x = x + alpha * (beta[index, :] - x)
        return (theta)


beta = np.load("../saved-outputs/beta.npy")
input_folder = "../input-data"
docs_file = f"{input_folder}/docs.txt"
wordids, wordcts = utilities.read_data(docs_file)
ml_ope = INFERENCE(14628, 15, 0.0666, beta, 50)
theta = ml_ope.static_online(20299, wordids, wordcts)
np.save("../saved-outputs/theta_mlope.npy", theta)