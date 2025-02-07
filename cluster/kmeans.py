import numpy as np
from scipy.spatial.distance import cdist


class KMeans:
    def __init__(self, k: int, tol: float = 1e-6, max_iter: int = 100):
        """
        In this method you should initialize whatever attributes will be required for the class.

        You can also do some basic error handling.

        What should happen if the user provides the wrong input or wrong type of input for the
        argument k?

        inputs:
            k: int
                the number of centroids to use in cluster fitting
            tol: float
                the minimum error tolerance from previous error during optimization to quit the model fit
            max_iter: int
                the maximum number of iterations before quitting model fit
        """

        self.k = k #initialize input variables
        self.tol = tol
        self.max_iter = max_iter
        #make sure that the input number of clusters is valid (positive nonzero integer)
        if k <= 0:
            raise ValueError("Number of clusters cannot be negative or 0")
        if type(k) is not int:
            raise ValueError("Number of clusters must be an integer")




    def fit(self, mat: np.ndarray):
        """
        Fits the kmeans algorithm onto a provided 2D matrix.
        As a bit of background, this method should not return anything.
        The intent here is to have this method find the k cluster centers from the data
        with the tolerance, then you will use .predict() to identify the
        clusters that best match some data that is provided.

        In sklearn there is also a fit_predict() method that combines these
        functions, but for now we will have you implement them both separately.

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features
        """

        matrix = mat

        k = self.k
        tol = self.tol
        max_iter = self.max_iter

        #we initialize k random centroid points based on the range of our data (starter_centroid), with k++

        self.centroid = np.zeros((k, matrix.shape[1]))

        initial_curr_centroid = np.random.uniform(np.min(matrix, axis=0), np.max(matrix, axis=0), size=(1, matrix.shape[1]))
        self.centroid[0] = initial_curr_centroid
 
        for i in range(1, k):
            total_dist = cdist(matrix, initial_curr_centroid)
            distribution_vals = (np.min(total_dist, axis=1))**2
            distribution = distribution_vals / np.sum(distribution_vals)
            self.centroid[i] = matrix[np.random.choice(matrix.shape[0], p=distribution)]

        curr_iter = 0

        while (curr_iter < max_iter):

            #compute all the MSE distances and get the index of the closest point for each centroid

            all_distances = cdist(matrix, self.centroid)

            self.mse = np.mean(all_distances**2)

            closest_points = np.argmin(all_distances, axis=1) 

            #group together points that are in the same cluster and calculate their centroid

            #got the nice dictionary format from ChatGPT, asked how to create a mapping (get indices of same value from list, group based on index assignment)
            #started from here https://stackoverflow.com/questions/70488053/how-to-get-the-indexes-of-the-same-values-in-a-list
            index_groups = {i: np.where(closest_points == i)[0] for i in range(k)}

            self.index_groups = index_groups

            new_centers = np.zeros((k, matrix.shape[1]))
            changes = []

            for i in range(k): #iterate through each cluster
                if len(matrix[index_groups[i]]) == 0: #we keep the same centroid if it is empty (no cells)
                    new_centers[i] = self.centroid[i]

                else:
                    mini_mat = matrix[index_groups[i]] #otherwise we calculate the new possible center for the cluster and the change in the center
                    my_center = np.mean(mini_mat, axis=0)
                    center_change = np.linalg.norm(self.centroid[i] - my_center)

                    new_centers[i] = my_center
                    changes.append(center_change)
         
            if np.max(changes) < tol: #if the max change in the cluster is less than the tolerance, we stop
                break
       
            self.centroid = new_centers #otherwise we update the centers

            curr_iter += 1

        

    def predict(self, mat: np.ndarray) -> np.ndarray:
        """
        Predicts the cluster labels for a provided matrix of data points--
            question: what sorts of data inputs here would prevent the code from running?
            How would you catch these sorts of end-user related errors?
            What if, for example, the matrix is of a different number of features than
            the data that the clusters were fit on?

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features

        outputs:
            np.ndarray
                a 1D array with the cluster label for each of the observations in `mat`
        """
        #check if the number of features matches, and return the array of indices that correspond to the closest centroid index for each point

        matrix = mat
        if matrix.shape[1] != len(self.centroid[0]):
            raise ValueError("The dimensions of the input matrix and centroid are not the same")
        
        all_distances = cdist(matrix, self.centroid)
        return np.argmin(all_distances, axis=1)


        

    def get_error(self) -> float:
        """
        Returns the final squared-mean error of the fit model. You can either do this by storing the
        original dataset or recording it following the end of model fitting.

        outputs:
            float
                the squared-mean error of the fit model
        """
        #we compute and retain MSE as part of the fit, here we return it
        return self.mse



    def get_centroids(self) -> np.ndarray:
        """
        Returns the centroid locations of the fit model.

        outputs:
            np.ndarray
                a `k x m` 2D matrix representing the cluster centroids of the fit model
        """
        #we compute and retain centroid as part of the fit, here we return it
        return self.centroid
