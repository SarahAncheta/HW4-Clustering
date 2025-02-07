# write your silhouette score unit tests here
#make sure to test against sklearn

from cluster.silhouette import Silhouette
from cluster.utils import (
        make_clusters, 
        plot_clusters,
        plot_multipanel)
from cluster.kmeans import KMeans
import numpy as np
import pytest


from sklearn.metrics import silhouette_score

def test_score_sil():
    kmeans = KMeans(k=3)
    clusters, _ = make_clusters(scale=0.3)
    kmeans.fit(clusters)
    y = kmeans.predict(clusters)

    sil = Silhouette()
    myscores = sil.score(clusters, y)
    #print(myscores.mean())
    sk_scores = silhouette_score(clusters, y)
    #print(sk_scores.mean())

    assert np.isclose(myscores.mean(),sk_scores.mean(), rtol=1e-03)

def test_inputs_sil():
    
    kmeans = KMeans(k=3)
    clusters, _ = make_clusters(scale=0.3)
    kmeans.fit(clusters)
    y = kmeans.predict(clusters)

    sil = Silhouette()

    with pytest.raises(ValueError):
        sil.score(clusters, np.append(y, 1))

    


