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


def test_fit_against_true_labels(): #make sure that the three predicted sets of indicies completely overlap with their true values
    k = 3
    kmeans = KMeans(k)
    clusters, true_labels = make_clusters(scale=0.3)
    kmeans.fit(clusters)

    predictions = kmeans.predict(clusters)

    #got this nice dictionary line from ChatGPT
    index_groups_true = {val: np.where(true_labels == val)[0].tolist() for val in np.unique(true_labels)}

    index_groups_predict = {val: np.where(predictions == val)[0].tolist() for val in np.unique(predictions)}

    #got this frozenset from ChatGPT
    sets1 = {frozenset(v) for v in index_groups_true.values()}
    sets2 = {frozenset(v) for v in index_groups_predict.values()}

    assert sets1 == sets2
    

    



