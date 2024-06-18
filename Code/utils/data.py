import os
import torch
import torchvision
import numpy as np
from collections import Counter
from sklearn.datasets import make_moons, make_blobs
from sklearn.preprocessing import scale
from PIL import Image


def get_data_path():
    return "./data"

class DatasetsSynthetic():
    """Simple enum class for storing data names"""
    TWOMOONS = "two-moons-plus-two-blobs"
    ALL = [TWOMOONS]

class DatasetsImages():
    """Simple enum class for storing data names"""
    MNIST = "mnist"
    FMNIST = "fashion_mnist"
    KMNIST = "kmnist"
    USPS = "usps"
    ALL = [MNIST, KMNIST, FMNIST, USPS]

class DatasetsUci():
    """Simple enum class for storing data names"""
    MICEPROTEIN = "mice_protein"
    PENDIGITS = "pendigits"
    HAR = "har"
    ALL = [MICEPROTEIN, HAR, PENDIGITS]

class AllData():
    ALL = DatasetsUci.ALL + DatasetsImages.ALL + DatasetsSynthetic.ALL
    

def print_data_statistics(data, labels=None):
    print()
    print("Data Set Information")
    print("Number of data points: ", data.shape[0])
    print("Number of dimensions: ", data.shape[1])
    print(f"Mean: {data.mean():.2f}, Standard deviation: {data.std():.2f}")
    print(f"Min: {data.min():.2f}, Max: {data.max():.2f}")
    if labels is not None:
        print("Number of classes: ", len(set(labels.tolist())))
        print("Class distribution:\n", sorted(Counter(labels.tolist()).items()))
    print()


def shuffle_dataset(x, y=None, subsample_size=None, return_index=False, random_state=None, replace=False, sampling_weights=None):
    if random_state is None:
        random_state = np.random.randint(100000)
    rng = np.random.default_rng(random_state)
    if subsample_size is None:
        subsample_size = x.shape[0]
    rand_idx = rng.choice(x.shape[0], subsample_size, replace=replace, p=sampling_weights)
    x_new = x[rand_idx]
    if y is not None:
        y_new = y[rand_idx]
        if return_index: return_vals = (x_new, y_new, rand_idx)
        else: return_vals = (x_new, y_new)
    else:
        if return_index: return_vals = (x_new, rand_idx)
        else: return_vals = x_new
    return return_vals

def generate_synthetic_example(N=2000):
    """Generates the two-moons-two-blobs aka SYNTH data set from the paper. 
       The intention of this data set was to illustrate how DECCS can improve the clustering results of each ensemble member
       (KM, SC, AGG, GMM) even in the case where none of them can cluster it perfectly in the beginning."""
    X, Y = make_moons(N, noise=0.04)
    X_blob, Y_blob = make_blobs(n_samples=N//2, n_features=2, cluster_std=[0.02, 0.045], centers=[ [2,.71], [2,1.0]])
    Y_blob += (Y.max()+1)
    
    X = np.concatenate([X,X_blob])
    Y = np.concatenate([Y,Y_blob])
    return torch.from_numpy(X).float(),torch.from_numpy(Y).long()

def load_synthetic(name, as_tensor=True, verbose=True):
    x_y = np.loadtxt(os.path.join(get_data_path(), name + ".txt"))
    data = x_y[:, :-1]
    labels = x_y[:, -1]
    if as_tensor:
        data = torch.from_numpy(data).float()
        labels = torch.from_numpy(labels).long()
    if verbose:
        print("Dataset: ",name)
        print_data_statistics(data, labels)
    return data, labels

def fetch_data_set_by_name(name, train=None, verbose=False):
    if name in DatasetsUci.ALL: 
        X, y = load_uci(name=name, verbose=verbose)
    elif name in DatasetsImages.ALL:
        if train is None:
            X, y = load_images(name=name, train=True, verbose=verbose)
            X_t, y_t = load_images(name=name, train=False, verbose=verbose)
            X = torch.cat([X, X_t])
            y = torch.cat([y, y_t])
        else:
            X, y = load_images(name=name, train=train, verbose=verbose)
    elif name in DatasetsSynthetic.ALL:
        X, y = load_synthetic(name=name, verbose=verbose)

    else:
        raise ValueError(f"Name: {name} is not available, has to be one of {AllData.ALL}")
    
    return X, y

def load_images(name, train=True, verbose=True):
    n = name.lower()
    if n == DatasetsImages.MNIST:
        X, y = load_mnist(train=train, verbose=verbose)
    elif n == DatasetsImages.FMNIST:
        X, y = load_fashion_mnist(train=train, verbose=verbose)
    elif n == DatasetsImages.KMNIST:
        X, y = load_kmnist(train=train, verbose=verbose)
    elif n == DatasetsImages.USPS:
        X, y = load_usps(train=train, verbose=verbose)
    return X, y

def load_usps(train=True, verbose=True):
    # setup normalization function
    mean = 0.2497
    std = 0.3010
    normalize = torchvision.transforms.Normalize((mean,), (std,))
    trainset = torchvision.datasets.USPS(root=get_data_path(), train=train, download=True)
    data = torch.from_numpy(trainset.data)

    # Scale to [0,1]
    data = data.float()/255
    # Apply z-transformation
    data = normalize(data)
    data = data.reshape(-1, data.shape[1] * data.shape[2])
    labels = torch.tensor(trainset.targets).long()
    if verbose:
        print("Dataset USPS")
        print_data_statistics(data, labels)
    return data, labels



def load_mnist(train=True, verbose=True):
    # setup normalization function
    mean = 0.1307
    std = 0.3081
    normalize = torchvision.transforms.Normalize((mean,), (std,))

    trainset = torchvision.datasets.MNIST(root=get_data_path(), train=train, download=True)
    data = trainset.data

    # Scale to [0,1]
    data = data.float()/255
    # Apply z-transformation
    data = normalize(data)
    # Flatten from a shape of (-1, 28,28) to (-1, 28*28)
    data = data.reshape(-1, data.shape[1] * data.shape[2])
    labels = trainset.targets
    if verbose:
        print("Dataset MNIST")
        print_data_statistics(data, labels)
    return data, labels

def load_fashion_mnist(train=True, verbose=True):
    # setup normalization function
    mean = 0.2860
    std = 0.3530
    normalize = torchvision.transforms.Normalize((mean,), (std,))
    trainset = torchvision.datasets.FashionMNIST(root=get_data_path(), train=train, download=True)
    data = trainset.data
    
    # Scale to [0,1]
    data = data.float()/255
    # Apply z-transformation
    data = normalize(data)
    data = data.reshape(-1, data.shape[1] * data.shape[2])
    labels = trainset.targets
    if verbose:
        print("Dataset FashionMNIST")
        print_data_statistics(data, labels)
    return data, labels


def load_kmnist(train=True, verbose=True):
    # setup normalization function
    mean = 0.19
    std = 0.35
    normalize = torchvision.transforms.Normalize((mean,), (std,))

    trainset = torchvision.datasets.KMNIST(root=get_data_path(), train=train, download=True)
    data = trainset.data
    # Scale to [0,1]
    data = data.float()/255
    # Apply z-transformation
    data = normalize(data)
    # Flatten from a shape of (-1, 28,28) to (-1, 28*28)
    data = data.reshape(-1, data.shape[1] * data.shape[2])
    labels = trainset.targets
    if verbose:
        print("Dataset KMNIST")
        print_data_statistics(data, labels)
    return data, labels



def load_uci(name, verbose=True):
    if name in DatasetsUci.ALL:
        if name in [DatasetsUci.PENDIGITS]:
            file_ending = ".csv"
            delimiter = ";"
            last_column_are_labels = False
        else:
            file_ending = ".txt"
            delimiter = ","
            last_column_are_labels = True
        data_path = os.path.join(get_data_path(), name + file_ending)
        data = np.loadtxt(data_path, delimiter=delimiter)
        if last_column_are_labels:
            X, y = data[:,:-1].astype(float), data[:,-1].astype(int)
        else:
            X, y = data[:,1:].astype(float), data[:,0].astype(int)

        # labels start from 1 and not from 0
        if y.min() == 1:
            y -= 1
        X = scale(X)
    else:
        raise ValueError(f"{name} is not in {DatasetsUci.ALL}")
    if verbose:
        print("Dataset: ",name)
        print_data_statistics(X, y)
    return torch.from_numpy(X).float(), torch.from_numpy(y).long()

class Dataset_with_indices(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return (len(self.data))

    def __getitem__(self, i):
        return self.data[i], i
