from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity 
import numpy as np

class Woodscore():
    """
    Computes the wood score for a given test set.

    Attributes:
    -----------
    xtrain : list
        The training data.
    xtest : list
        The test data.
    ytest : list
        The reference labels for the test data.
    pred : list
        The predicted labels for the test data.
    model : sentence_transformers.SentenceTransformer
        The model to be used for encoding the training and test data: https://www.sbert.net/#
    a : float
        A parameter that controls the weight of p in the final score.
    b : int
        The number of training samples with highest similarity scores to be used for computing p.

    Algorithm:
    ----------
    1. Compute the cosine similarity matrix between the test and training data
        which returns a (n_test_samples, n_test_sample) sized array.
    2. Compute the summed matrix using the cosine similarity matrix
        only usined the first "b" number of test cases ordered in
        descending order by simularity score.
    3. Compute p for each test sample using the summed matrix and the parameter a.
    4. Compute the accuracy score for each test sample.
    5. Compute the wood score by taking the weighted average of the accuracy scores.

    Example:
    ----------
    from sentence_transformers import SentenceTransformer, util
    from sklearn.datasets import fetch_20newsgroups
    import numpy as np
    from woodscore import Woodscore

    model = SentenceTransformer('all-MiniLM-L6-v2')

    cats = ['alt.atheism', 'sci.space']
    newsgroups_train = fetch_20newsgroups(subset='train', categories=cats)

    newsgroups_test = fetch_20newsgroups(subset='test',
                                    categories=cats)

    b = 50

    a = 10

    predictions = np.zeros((len(newsgroups_test.target),))

    wood_score = Woodscore(newsgroups_train.data, newsgroups_test.data,
                                    newsgroups_test.target, predictions, model, b, a)
                                   
    wood_score.compute_metric()

    Returns:
    --------
    result : float
        The wood score.
    """
    def __init__(self, training_data, test_data, references, predictions, model, b, a):
        self.xtrain = training_data
        self.xtest = test_data
        self.ytest = references
        self.pred = predictions
        self.model = model
        self.a = a
        self.b = b

    def compute_cosine_similatiry(self):
        embeddings_xtrain = self.model.encode(self.xtrain, convert_to_tensor=True).cpu()
        embedding_xtest = self.model.encode(self.xtest, convert_to_tensor=True).cpu()
        cosine_sim_matrix = cosine_similarity(embedding_xtest, embeddings_xtrain)
        return cosine_sim_matrix

    def compute_summed_matrix(self, cosine_sim_matrix):
        sorted_matrix = -np.sort(-cosine_sim_matrix)
        sum_matrix = np.sum(cosine_sim_matrix[:, :self.b], axis = 1)
        return sum_matrix

    def compute_metric(self):
        cosine_sim_matrix = self.compute_cosine_similatiry()
        sum_matrix = self.compute_summed_matrix(cosine_sim_matrix)
        p_matrix = self.a / sum_matrix

        accuracies = []
        for count, value in enumerate(self.ytest):
            accuracy = accuracy_score([value], [int(self.pred[count])])
            p = p_matrix[count]
            accuracies.append(accuracy*p)

        result = sum(accuracies) / len(accuracies)
        print(f'Wood Score v1 = {result}!')
        return result

