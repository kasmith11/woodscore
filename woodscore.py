from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity 
import numpy as np

class Woodscore():
    """
    Step 1 - For the init what do we need from the user? - Parameters??

    "I" looks like an initial parameter?
    """
    def __init__(self, training_data, test_data, references, predictions, model, b, a):
        self.accuracy = []
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
        """Returns the scores
        
        Algorithm:

        https://www.sbert.net/#

        We want to take the cosine similarity between each sample in the training set with each sample of the
        testing set.

        To get rid of a for loop we can run use a matrix for the cosine similarity!  
                https://stackoverflow.com/questions/50411191/how-to-compute-the-cosine-similarity-in-pytorch-for-all-rows-in-a-matrix-with-re  
        
        So now this is all a function of how we'd like to take the mean of that matrix?:
            https://pytorch.org/docs/stable/generated/torch.mean.html#torch.mean
            It should be accross the first axis - torch.mean(a, 1)

        Once we have all of our average similarities we can enumerate over all of the test cases?:
            https://www.geeksforgeeks.org/use-enumerate-and-zip-together-in-python/
        
        
        """
        # TODO: Compute the different scores of the modules
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

