import numpy as np
from scipy.spatial.distance import cdist


class Silhouette:
    def __init__(self):
        """
        inputs:
            none
        """

    def score(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        calculates the silhouette score for each of the observations

        inputs:
            X: np.ndarray
                A 2D matrix where the rows are observations and columns are features.

            y: np.ndarray
                a 1D array representing the cluster labels for each of the observations in `X`

        outputs:
            np.ndarray
                a 1D array with the silhouette scores for each of the observations in `X`
        """

        if X.shape[0] != len(y):
            raise ValueError("The number of items in the input matrix and number of cluster labels are not the same")
        

        #create a dictionary for the indices that correspond to each cluster

        sub_indices_per_cluster = {}
        
        for i in np.unique(y): #we iterate through each of the clusters and get indices for each
            cluster_indices = np.where(y == i)[0]
            sub_indices_per_cluster[i] = cluster_indices

        scores = np.zeros(X.shape[0])

        for cell in range(X.shape[0]):
            mycell = X[cell]
            mycluster = y[cell]
            
            #get the indices for cells that are in my cluster, excluding me (current cell)

            inter_indices = np.where(y == mycluster)[0]
            inter_indices = np.delete(inter_indices, np.where(inter_indices == cell))
            
            #calculate inter cluster coherence

            inter_neighbor = X[inter_indices]

            inter_coh = np.mean(np.linalg.norm(inter_neighbor - mycell, axis=1))

            #calculate intra cluster coherence for all clusters
            intra_options = []

            #iterate through each cluster
            for k, v in sub_indices_per_cluster.items():
                if k != mycluster:
                    intra_options.append(np.mean(np.linalg.norm(X[v] - mycell, axis=1)))
                else:
                    continue
            #get the cluster with the minimum coherence
            intra_coh = np.min(intra_options)

            #calculate final score
            cell_score = (inter_coh - intra_coh)/np.max(inter_coh, intra_coh)
            scores[cell] = cell_score

        return scores

            
            


            

            




