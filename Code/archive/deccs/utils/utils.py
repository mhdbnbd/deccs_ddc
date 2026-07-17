# General utility functions

import os
import torch
import torchvision
import numpy as np
import json
import random
from math import prod
from itertools import islice
from scipy.optimize import linear_sum_assignment
import pandas as pd

def flatten_tensor(x):
    return x.reshape(-1, prod(x.shape[1:]))

def random_seed(seed):
    """Used for debugging. Fixes randomness for necessary libraries"""
    random.seed(seed)
    np.random.seed(random.randint(0, 10000))
    torch.manual_seed(random.randint(0, 10000))
    torch.cuda.manual_seed_all(random.randint(0, 10000))
    torch.backends.cudnn.deterministic = True


def setup_directory(dir_name, verbose=False):
    """
    Setup directory in case it does not exist
    Parameters:
    -------------
    dir_name: str, path + name to directory
    verbose: bool, indicates whether directory creation should be printed or not.
    """
    if not os.path.exists(dir_name):
        try:
            os.makedirs(dir_name)
            if verbose:
                print("Created Directory: {}".format(dir_name))
        except Exception as e:
            raise RuntimeError(
                "Could not create directory: {}\n {}".format(dir_name, e))

def squared_euclidean_distance(centers, embedded, weights=None):
    ta = centers.unsqueeze(0)
    tb = embedded.unsqueeze(1)
    squared_diffs = (ta - tb)
    if weights is not None:
        weights_unsqueezed = weights.unsqueeze(0).unsqueeze(1)
        squared_diffs = squared_diffs * weights_unsqueezed
    squared_diffs = squared_diffs.pow(2).mean(2)
    
    return squared_diffs

def discretize_in_bins(x):
    """Discretize a vector in two bins."""
    x_min = x.min(1)[0]
    x_max = x.max(1)[0]
    thresh = (0.5 *(x_min + x_max)).unsqueeze(1)
    mask = x > thresh
    return mask, thresh

def int_to_one_hot(label_tensor, n_labels):
    """
    Creates a one_hot_matrix for the given label vector.
    """
    onehot = torch.zeros([label_tensor.shape[0], n_labels]).to(label_tensor.device)
    onehot.scatter_(1, label_tensor.unsqueeze(1), 1.0)
    return onehot

def detect_device(which_device=0):
    """Automatically detects if you have a cuda enabled GPU"""
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{which_device}')
    else:
        device = torch.device('cpu')
    return device

def encode_batchwise(dataloader, model, device):
    """ Utility function for embedding the whole data set in a mini-batch fashion
    """
    with torch.no_grad():
        model.eval()
        embeddings = []
        for batch in dataloader:
            batch_data = batch[0].to(device)
            embeddings.append(model.encode(batch_data).detach().cpu())
    return torch.cat(embeddings, dim=0).numpy()

def set_dropout_train(model):
    def set_dropout(m):
        classname = m.__class__.__name__
        if classname.find("Dropout") != -1:
            m.train()        
    model.apply(set_dropout)
    return model

def encode_batchwise_from_data(data, model, device, batch_size=128):
    """ Utility function for embedding the whole data set in a mini-batch fashion
    """
    with torch.no_grad():
        model.eval()
        ds = torch.utils.data.TensorDataset(*(data, ))
        dataloader = torch.utils.data.DataLoader(ds,
                                                batch_size=batch_size,
                                                shuffle=False,
                                                drop_last=False)
        embeddings = []
        for batch in dataloader:
            batch_data = batch[0].to(device)
            emb_i = model.encode(batch_data)
            if isinstance(emb_i, tuple):
                emb_i = emb_i[0]
            emb_i = emb_i.detach().cpu()
            embeddings.append(emb_i)
    return torch.cat(embeddings, dim=0).numpy()


def decode_batchwise(dataloader, model, device):
    """ Utility function for decoding the whole data set in a mini-batch fashion
    """
    decodings = []
    for batch in dataloader:
        batch_data = batch[0].to(device)
        decodings.append(model(batch_data).detach().cpu())
    return torch.cat(decodings, dim=0).numpy()

def predict_batchwise(dataloader, model, cluster_module, device):
    """ Utility function for predicting the cluster labels over the whole data set in a mini-batch fashion
    """
    with torch.no_grad():
        predictions = []
        for batch in dataloader:
            batch_data = batch[0].to(device)
            if model is None:
                prediction = cluster_module.prediction_hard(batch_data).detach().cpu()
            else:
                z = model.encode(batch_data)
                prediction = cluster_module.prediction_hard(z).detach().cpu()
            predictions.append(prediction)
    return torch.cat(predictions, dim=0).numpy()



def evaluate_batchwise(dataloader, model, cluster_module, device):
    """ Utility function for evaluating the cluster performance with NMI in a mini-batch fashion
    """
    predictions = []
    labels = []
    for batch in dataloader:
        batch_data = batch[0].to(device)
        label = batch[1]
        labels.append(label)
        prediction = cluster_module.prediction_hard(model.encode(batch_data)).detach().cpu()
        predictions.append(prediction)
    predictions = torch.cat(predictions, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy()
    return normalized_mutual_info_score(labels, predictions)

def write_json(fname, dictionary):
    """
    This function saves a dictionary as json file to disk.

    Parameters:
    ---------------
    fname: str, name or path+name of file
    dictionary: dict, which should be saved to disk

    """
    json_path = os.path.join(fname)
    with open(json_path, 'w') as outfile:
        json.dump(dict(dictionary), outfile, indent=4, sort_keys=True)

def load_json(file_path):
    """
    This function loads a json as dictionary from disk.

    Parameters:
    ---------------
    file_path: str, name or path+name of file
    Returns:
    ---------------
    dictionary: dict, with content of json file.
    """
    try:
        dictionary = json.load(open(file_path, "r"))
    except Exception as e:
        print('Json value expected: ', e)
    return dictionary


def window(seq, n):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result


def cluster_accuracy(y_true, y_pred):
    """
    Calculate clustering accuracy with the hungarian method.
    code adapted from:
    https://github.com/XifengGuo/IDEC/blob/master/datasets.py
    Parameters:
    -----------------
    y: true labels, numpy.array with shape `(n_samples,)`
    y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    Returns:
    -----------------
    accuracy, in [0,1]
    Raises:
    -----------------
    ValueError: If size of y_true and y_pred do not match
    """
    if isinstance(y_true, list):
        y_true = np.array(y_true)
    if isinstance(y_pred, list):
        y_pred = np.array(y_pred)
    y_true = y_true.astype(np.int64)
    if(y_pred.size != y_true.size):
        raise ValueError("y_true and y_pred sizes do not match, they are: {} != {}".format(
            y_pred.size, y_true.size))
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)

    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    # apply hungarian method
    ind = linear_sum_assignment(w.max() - w)
    ind = np.array(ind).transpose()
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

# ddc utils

def load_data(data_path):
    # Load the dataset
    data = pd.read_csv(data_path)
    return data.values

def load_tags(tags_path):
    # Load the tags dataset
    tags = pd.read_csv(tags_path)
    return tags.values

def evaluate_clustering(data, clusters, explanations, pairwise_loss):
    # Implement evaluation logic
    pass