from copy import deepcopy
from collections import OrderedDict, Counter
from sklearn.preprocessing import scale
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
import torch
from utils.utils import predict_batchwise
from utils.utils import int_to_one_hot
import numpy as np
from utils.data import shuffle_dataset

def calculate_avg_agreement(agreement_dict):
    avg_agreement = 0
    for name, value in agreement_dict.items():
        avg_agreement += agreement_dict[name]
    avg_agreement /= len(agreement_dict)
    return avg_agreement

def get_assignment_matrix_dict(cluster_dict, device=torch.device("cpu")):
    assignment_matrix_dict = OrderedDict()
    for name, labels in cluster_dict.items():
        n_clusters = len(set(labels.tolist()))
        max_label = np.max(labels)
        if n_clusters < (max_label + 1):
            raise ValueError(f"{name} classifier lost clusters. Expected {max_label+1}, but found only {n_clusters}")
        assignment_matrix_dict[name] = int_to_one_hot(torch.from_numpy(labels).long(), n_clusters).detach().to(device)
    return assignment_matrix_dict

def get_prediction_dict(classifier_dict, model, dataloader, device=torch.device("cpu"), labels=None):
    prediction_dict = OrderedDict()
    if isinstance(classifier_dict, torch.nn.Module):
        items = classifier_dict.classifiers.items()
    elif isinstance(classifier_dict, dict):
        items = classifier_dict.items()
    for name, c_i in items:
        pred_y = predict_batchwise(dataloader, model, c_i, device)
        prediction_dict[name] = pred_y
        if labels is not None:
            nmi = normalized_mutual_info_score(labels, pred_y)
            ari = adjusted_rand_score(labels, pred_y)
            print(f"{name} NMI: {nmi:.4f} ARI: {ari:.4f}")
    return prediction_dict

def eval_prediction_dict(prediction_dict, metrics_fn_dict, labels, metric_suffix="", res_dict=None):
    if res_dict is None:
        res_dict = OrderedDict({name:OrderedDict() for name, _ in prediction_dict.items()})
        for name, _ in res_dict.items():
            for metric_name, _ in metrics_fn_dict.items():
                res_dict[name][metric_name+metric_suffix] = []
    for name, pred_y in prediction_dict.items():
        for metric_name, metric_i in metrics_fn_dict.items():
            res_dict[name][metric_name+metric_suffix].append(metric_i(labels, pred_y))
    return res_dict

def get_soft_prediction_dict(classifier_dict, model, dataloader, device=torch.device("cpu"), labels=None):
    def soft_predict_batchwise(dataloader, model, cluster_module, device):
        predictions = []
        for batch in dataloader:
            batch_data = batch[0].to(device)
            prediction = cluster_module(model.encode(batch_data)).detach().cpu()
            predictions.append(prediction)
        return torch.cat(predictions, dim=0).numpy()

    prediction_dict = OrderedDict()
    if isinstance(classifier_dict, torch.nn.Module):
        items = classifier_dict.classifiers.items()
    elif isinstance(classifier_dict, dict):
        items = classifier_dict.items()
    for name, c_i in items:
        pred_y = soft_predict_batchwise(dataloader, model, c_i, device)
        prediction_dict[name] = pred_y
    return prediction_dict

def get_classifier_uncertainty_weights(classifier_dict, model, dataloader, device=torch.device("cpu"), past_weights=None):
    soft_prediction_dict = get_soft_prediction_dict(classifier_dict=classifier_dict,
                                            model=model,
                                            dataloader=dataloader,
                                            device=device,
                                            labels=None)
    w = None
    # sum over maximum predictions for each data point per classifier
    for _, arr in soft_prediction_dict.items():
        max_i = arr.max(axis=1)
        if w is None:
            w = max_i
        else:
            w += max_i
    # average over classifiers
    w /= len(soft_prediction_dict)
    # inverse weighting because we want to sample uncertain data points more often
    w = 1/(w+1e-8)
    # use running average to account for past uncertainties
    if past_weights is not None:
        w = 0.5*w + 0.5*past_weights
    # scale to be between 0 and 1
    w /= w.sum()
    return w

def filter_best_predictions(soft_prediction_dict, threshold, data, labels, verbose=True):
    mask = None
    for name, soft_pred_i in soft_prediction_dict.items():
        if mask is None:
            mask = soft_pred_i > threshold
        else:
            mask += soft_pred_i > threshold
        at_least_one_mask = mask.sum(1) > 0
    if verbose:
        print(f"Filtered {data[at_least_one_mask,:].shape[0]} from {at_least_one_mask.shape[0]}")
    return data[at_least_one_mask,:], labels[at_least_one_mask]
    

def get_clusterer_dict(n_clusters):
    clusterer_dict = OrderedDict({"KMeans": KMeans(n_clusters=n_clusters),
                                "SpectralClustering": SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors',n_neighbors=10, assign_labels='kmeans'),
                                "AgglomerativeClustering": AgglomerativeClustering(n_clusters=n_clusters),
                             })
    return cluster_dict


def _split_cluster_dict(data, cluster_dict, n_train, gt_labels=None, subsample_idx=None, eval_subsample_idx_to_delete=None):
    cluster_dict_train = OrderedDict({})
    cluster_dict_test = OrderedDict({})
    for name, labels in cluster_dict.items():
        if subsample_idx is None:
            subsample_size = int(n_train * labels.shape[0])
            _, subsample_idx = shuffle_dataset(labels, subsample_size=subsample_size, return_index=True)
            
        cluster_dict_train[name] = labels[subsample_idx]
        if eval_subsample_idx_to_delete is None:
            cluster_dict_test[name] = np.delete(labels, subsample_idx)
        else:
            cluster_dict_test[name] = np.delete(labels, eval_subsample_idx_to_delete)

    data_train = data[subsample_idx]
    if eval_subsample_idx_to_delete is None:
        data_test = np.delete(data, subsample_idx, axis=0)
    else:
        data_test = np.delete(data, eval_subsample_idx_to_delete, axis=0)

    if gt_labels is None:
        return cluster_dict_train, cluster_dict_test, data_train, data_test, subsample_idx
    else:
        gt_labels_train = gt_labels[subsample_idx]
        if eval_subsample_idx_to_delete is None:
            gt_labels_test = np.delete(gt_labels, subsample_idx, axis=0)
        else:
            gt_labels_test = np.delete(gt_labels, eval_subsample_idx_to_delete, axis=0)

        return cluster_dict_train, cluster_dict_test, data_train, data_test, gt_labels_train, gt_labels_test, subsample_idx

def cluster_dict_train_test_split(data, cluster_dict, n_train, gt_labels=None, equal_split=True):
    if gt_labels is None:
        cluster_dict_train, cluster_dict_test, data_train, data_test, subsample_idx = _split_cluster_dict(data, cluster_dict, n_train, gt_labels)
    else:
        cluster_dict_train, cluster_dict_test, data_train, data_test, gt_labels_train, gt_labels_test, subsample_idx = _split_cluster_dict(data, cluster_dict, n_train, gt_labels)
    if equal_split:
        indices_train = []
        indices_test = []
        for name, labels in cluster_dict.items():
            unique_train_labels = set(cluster_dict_train[name].tolist())
            unique_test_labels = set(cluster_dict_test[name].tolist())
            diff_train = unique_train_labels.difference(unique_test_labels)
            diff_test = unique_test_labels.difference(unique_train_labels)
            diff = diff_train.union(diff_test)
            if len(diff) > 0:
                for l_i in list(diff):
                    indices_train += np.where(cluster_dict[name]==l_i)[0].tolist()
        # filter for unique idx
        indices_train = set(indices_train)
        # adjust subsample_idx
        subsample_idx = set(subsample_idx.tolist())
        subsample_idx = subsample_idx.union(indices_train)
        # duplicate train samples
        eval_subsample_idx_to_delete = subsample_idx.difference(indices_train)
        # Convert back to list
        subsample_idx = list(subsample_idx)
        eval_subsample_idx_to_delete = list(eval_subsample_idx_to_delete)
        # Split cluster_dict again with adjusted subsample_idx
        if gt_labels is None:
            cluster_dict_train, cluster_dict_test, data_train, data_test, _ = _split_cluster_dict(data, cluster_dict, n_train, gt_labels, subsample_idx=subsample_idx, eval_subsample_idx_to_delete=eval_subsample_idx_to_delete)
        else:
            cluster_dict_train, cluster_dict_test, data_train, data_test, gt_labels_train, gt_labels_test, _ = _split_cluster_dict(data, cluster_dict, n_train, gt_labels, subsample_idx=subsample_idx, eval_subsample_idx_to_delete=eval_subsample_idx_to_delete)
    

    if gt_labels is None:
        return cluster_dict_train, cluster_dict_test, data_train, data_test
    else:
        return cluster_dict_train, cluster_dict_test, data_train, data_test, gt_labels_train, gt_labels_test

def find_small_clusters(cluster_dict, threshold):
    sc_indices = []
    for name, labels_i in cluster_dict.items():
        unique_labels = list(set(labels_i.tolist()))
        c = Counter(labels_i)
        small_labels = [i for i in unique_labels if c[i] <= threshold]
        for label_i in small_labels:
            indices = np.where(labels_i==label_i)[0]
            sc_indices += indices.tolist()
    return sc_indices

def remove_small_clusters(cluster_dict, data, labels=None, threshold=5):
    updated_cluster_dict = deepcopy(cluster_dict)
    data_updated = deepcopy(data)
    if labels is not None:
        labels_updated = deepcopy(labels)
    # Find small clusters
    indices_to_remove = find_small_clusters(cluster_dict, threshold)
    # Remove small clusters
    while(indices_to_remove):
        for name, labels_i in updated_cluster_dict.items():
            labels_new = np.delete(labels_i, indices_to_remove, axis=0)
            # Renumber to make sure that labeling is coherent
            for enumerate_label, label_new_i in enumerate(set(labels_new)):
                # keep noise point label
                if label_new_i != -1:
                    labels_new[labels_new==label_new_i] = enumerate_label
            updated_cluster_dict[name] = labels_new

        data_updated = np.delete(data_updated, indices_to_remove, axis=0)
        if labels is not None:
            labels_updated = np.delete(labels_updated, indices_to_remove, axis=0)
        # Find small clusters in the updated cluster dict
        indices_to_remove = find_small_clusters(updated_cluster_dict, threshold)
    if labels is not None:
        return updated_cluster_dict, data_updated, labels_updated 
    else:
        return updated_cluster_dict, data_updated

def create_cluster_dict(clusterer_dict, X, labels=None, to_scale=True, verbose=False):
    cluster_dict = OrderedDict()
    if to_scale:
        X = scale(X)
    for name, clusterer_i in clusterer_dict.items():
        pred_i = clusterer_i.fit_predict(X)
        if verbose: print("Class distribution:\n", sorted(Counter(pred_i.tolist()).items()))        
        if labels is not None:
            nmi = normalized_mutual_info_score(labels, pred_i)
            ari = adjusted_rand_score(labels, pred_i)
            if verbose: print(f"{name} NMI: {nmi:.4f} ARI: {ari:.4f}")
        cluster_dict[name] = pred_i
    return cluster_dict


def calculate_weights(prediction_dict):
    weight_dict = OrderedDict({name:0 for name in list(prediction_dict.keys())})
    n_predictors = len(prediction_dict)
    for name_i, pred_i in prediction_dict.items():
        for name_j, pred_j in prediction_dict.items():
            if name_i != name_j:
                nmi_j = normalized_mutual_info_score(pred_i, pred_j)
                weight_dict[name_i] += nmi_j
        weight_dict[name_i] /= (n_predictors-1)
    return weight_dict

def calculate_max_normalized_weights(prediction_dict, max_diff=1e-2):
    weight_dict = OrderedDict({name:0 for name in list(prediction_dict.keys())})
    n_predictors = len(prediction_dict)
    highest_diff = 1
    for name_i, pred_i in prediction_dict.items():
        nmis = []
        for name_j, pred_j in prediction_dict.items():
            if name_i != name_j:
                nmi_j = normalized_mutual_info_score(pred_i, pred_j)
                weight_dict[name_i] += nmi_j
                nmis.append(nmi_j)
        max_agreement = max(nmis)
        diff = (1-max_agreement)
        if diff < max_diff:
            diff = max_diff
        weight_dict[name_i] /= (diff)*(n_predictors-1)
        if highest_diff > diff:
            highest_diff = diff
    # renormalize to be between 0 and 1
    for name_i, _ in prediction_dict.items():
        weight_dict[name_i] /= (1/highest_diff)
    return weight_dict
