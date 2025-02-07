# Write your k-means unit tests here


from cluster.kmeans import KMeans
from cluster.utils import (
        make_clusters, 
        plot_clusters,
        plot_multipanel)

import pytest
import numpy as np


def test_init_kmeans(): #check to make sure that we are only allowing positive nonzero k values

    with pytest.raises(ValueError):
        KMeans(k =-2)

    with pytest.raises(ValueError):
        KMeans(k =0)

    with pytest.raises(ValueError):
        KMeans(k = 1.5)


def test_fit_kmeans(): 

    kmeans = KMeans(k=1000)
    clusters, _ = make_clusters(scale=0.3)

    with pytest.raises(ValueError): #we want more clusters than points
        kmeans.fit(clusters)

def test_predict_kmeans():

    kmeans = KMeans(k=3)
    clusters, _ = make_clusters(scale=0.3)
    kmeans.fit(clusters)

    assert(isinstance(kmeans.predict(clusters), np.ndarray))


def test_mse_kmeans():

    kmeans = KMeans(k=3)
    clusters, _ = make_clusters(scale=0.3)
    kmeans.fit(clusters)

    assert(isinstance(kmeans.get_error(), float))

def test_centroids_kmeans():

    k = 3

    kmeans = KMeans(k)
    clusters, _ = make_clusters(scale=0.3)
    kmeans.fit(clusters)

    center = kmeans.get_centroids()

    assert(isinstance(kmeans.get_error(), float))
    assert center.shape[0] == k
    assert center.shape[1] == clusters.shape[1]




    



