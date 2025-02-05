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

        #we initialize k random centroid points based on the range of our data (starter_centroid)

        self.centroid = np.random.uniform(np.min(matrix, axis=0), np.max(matrix, axis=0), size=(k, matrix.shape[1]))

        curr_iter = 0

        while (curr_iter < max_iter):

            all_distances = cdist(matrix, self.centroid)

            self.mse = np.avg(all_distances**2)

            closest_points = np.argmin(all_distances, axis=1)

            #group together points that are in the same cluster and calculate their centroid

            #got the nice dictionary format from ChatGPT, asked how to create a mapping (get indices of same value from list, group based on index assignment)
            #started from here https://stackoverflow.com/questions/70488053/how-to-get-the-indexes-of-the-same-values-in-a-list
            index_groups = {i: np.where(closest_points == i)[0] for i in range(k)}

            self.index_groups = index_groups

            new_centers = np.zeros((k, matrix.shape[1]))

            for i in range(k):
                if len(matrix[index_groups[i]]) == 0: #we keep the same centroid if it is empty
                    new_centers[i] = self.centroid[i]

                else:
                    mini_mat = matrix[index_groups[i]]
                    my_center = np.mean(mini_mat, axis=0)
                    center_change = np.linalg.norm(self.centroid[i], my_center)

                    if center_change <  tol:
                        new_centers[i] = self.centroid[i]

                    else:
                        new_centers[i] = my_center

            if np.allclose(self.centroid, new_centers, tol): #this means that all were not updated, all passed the tolerance threshold
                break
            else:
                self.centroid = new_centers

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

        matrix = mat
        if len(matrix.shape[1]) != len(self.centroid[0]):
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
        return self.mse



    def get_centroids(self) -> np.ndarray:
        """
        Returns the centroid locations of the fit model.

        outputs:
            np.ndarray
                a `k x m` 2D matrix representing the cluster centroids of the fit model
        """

        return self.centroid
