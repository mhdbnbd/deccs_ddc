from collections import OrderedDict
import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture


def get_clusterer_diverse_dict(n_clusters):
    clusterer_dict = OrderedDict({"KM": KMeans(n_clusters=n_clusters, n_init=10),
                                  "SC": SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', n_neighbors=10, assign_labels='kmeans', n_init=10),
                                  "AGG": AgglomerativeClustering(n_clusters=n_clusters, linkage="ward"),
                                  "GMM": GaussianMixture(n_components=n_clusters, covariance_type="full", n_init=10, reg_covar=1e-5),
                                 })
    return clusterer_dict


def get_clusterer_2timesdiverse_dict(n_clusters):
    clusterer_dict = OrderedDict({
                                  "KM": KMeans(n_clusters=n_clusters, n_init=10),
                                  "KM2": KMeans(n_clusters=n_clusters, n_init=10),
                                  "SC": SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', n_neighbors=10, assign_labels='kmeans', n_init=10),
                                  "SC2": SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', n_neighbors=10, assign_labels='kmeans', n_init=10),
                                  "AGG": AgglomerativeClustering(n_clusters=n_clusters, linkage="ward"),
                                  "AGG2": AgglomerativeClustering(n_clusters=n_clusters, linkage="ward"),
                                  "GMM": GaussianMixture(n_components=n_clusters, covariance_type="full", n_init=10, reg_covar=1e-5),
                                  "GMM2": GaussianMixture(n_components=n_clusters, covariance_type="full", n_init=10, reg_covar=1e-5),
                                 })
    return clusterer_dict

def get_clusterer_3timesdiverse_dict(n_clusters):
    clusterer_dict = OrderedDict({
                                  "KM": KMeans(n_clusters=n_clusters, n_init=10),
                                  "KM2": KMeans(n_clusters=n_clusters, n_init=10),
                                  "KM3": KMeans(n_clusters=n_clusters, n_init=10),
                                  "SC": SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', n_neighbors=10, assign_labels='kmeans', n_init=10),
                                  "SC2": SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', n_neighbors=10, assign_labels='kmeans', n_init=10),
                                  "SC3": SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', n_neighbors=10, assign_labels='kmeans', n_init=10),
                                  "AGG": AgglomerativeClustering(n_clusters=n_clusters, linkage="ward"),
                                  "AGG2": AgglomerativeClustering(n_clusters=n_clusters, linkage="ward"),
                                  "AGG3": AgglomerativeClustering(n_clusters=n_clusters, linkage="ward"),
                                  "GMM": GaussianMixture(n_components=n_clusters, covariance_type="full", n_init=10, reg_covar=1e-5),
                                  "GMM2": GaussianMixture(n_components=n_clusters, covariance_type="full", n_init=10, reg_covar=1e-5),
                                  "GMM3": GaussianMixture(n_components=n_clusters, covariance_type="full", n_init=10, reg_covar=1e-5),
                                 })
    return clusterer_dict

def get_clusterer_diverse_synth_dict(n_clusters):
    clusterer_dict = OrderedDict({"KM": KMeans(n_clusters=n_clusters),
                                  "SC": SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors',n_neighbors=10, assign_labels="kmeans"),
                                  "AGG": AgglomerativeClustering(n_clusters=n_clusters, linkage="single"),
                                  "GMM": GaussianMixture(n_components=n_clusters, covariance_type="full", n_init=10, reg_covar=1e-5),
                                 })
    return clusterer_dict


class ClustererEnsembles():
    DICT = OrderedDict({
        "DIVERSE": get_clusterer_diverse_dict,
        "2xDIVERSE": get_clusterer_2timesdiverse_dict,
        "3xDIVERSE": get_clusterer_3timesdiverse_dict,
        "DIVERSE_SYNTH": get_clusterer_diverse_synth_dict,

    })
    ALL = ["DIVERSE", "2xDIVERSE", "3xDIVERSE", "DIVERSE_SYNTH"]