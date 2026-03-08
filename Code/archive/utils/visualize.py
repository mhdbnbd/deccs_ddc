# Utils for visualizing our results

import os
import torch
import torchvision
import numpy as np
import matplotlib 
from collections import Counter
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from matplotlib.colors import ListedColormap
import seaborn as sns
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment

def match_labels(y, gt_labels):
    """Matches order of labels y with gt_labels for visualization"""
    D = max(y.max(), gt_labels.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for j in range(y.size):
        w[y[j], gt_labels[j]] += 1
    # apply hungarian method to match colors
    ind = linear_sum_assignment(w.max() - w)
    ind = np.array(ind).transpose()
    ind_d = {i:j for i,j in ind}
    new_y = np.zeros_like(y)
    for j in range(y.shape[0]):
        new_y[j] = ind_d[y[j]]
    return new_y

def denormalize(tensor:torch.Tensor, mean:float=0.1307, std:float=0.3081, shape=(28,28))->torch.Tensor:
    """
    This applies an inverse z-transformation and reshaping to visualize the mnist images properly.
    """
    pt_std = torch.as_tensor(std, dtype=torch.float32, device=tensor.device)
    pt_mean = torch.as_tensor(mean, dtype=torch.float32, device=tensor.device)
    return (tensor.mul(pt_std).add(pt_mean).view(-1, 1, shape[0], shape[1]) * 255).int().detach()
 

def pca_plot(data, labels=None, figsize=(10,10), s=1, title=None):
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(data)
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(reduced_data[:,0], reduced_data[:,1], c=labels, s=s)
    if title is not None:
        ax.set_title(title)
    plt.show()

def plot_clusterings(X, prediction_dict, labels, figsize=(20,3), s=4, save=None, center_dict=None, cmap=None, lw=0.1):
    if X.shape[1] > 2:
        pca = PCA(n_components=2)
        X = pca.fit_transform(X)
    else:
        pca = None
    if cmap is None:
        cmap = plt.cm.RdBu
    n_subplots = len(prediction_dict.keys()) + 1
    fig, axes = plt.subplots(1,n_subplots, figsize=figsize)
    i = 0
    for name, pred_i in prediction_dict.items():
        ari = adjusted_rand_score(pred_i, labels)
        pred_i = match_labels(pred_i.astype(int), labels.astype(int))

        axes[i].scatter(X[:,0], X[:,1], c=pred_i, s=s, cmap=cmap, edgecolors="white", linewidths=lw)
        if center_dict is not None:
            centers = center_dict[name]
            if isinstance(centers, torch.Tensor):
                centers = centers.detach().cpu().numpy()
            if pca is not None:
                centers = pca.transform(centers)
            axes[i].scatter(centers[:,0], centers[:,1], c="black", s=100)

        axes[i].set_title(name + f" ARI:{ari:.3f}")
        i += 1
    axes[-1].scatter(X[:,0], X[:,1], c=labels, s=s, cmap=cmap, edgecolors="white", linewidths=lw)
    axes[-1].set_title("Ground Truth")
    if save is not None:
        plt.savefig(save)
    plt.show()


def plot_classifier_probabilities(classifier_dict, X, gt_labels=None, device=torch.device("cpu"), save=None,figsize=(14, 3), titlesize=20.5, plot_centers=True, s=45):
    # adapted from here: https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
    h = 0.01 # step size in the mesh
    plt.rc('axes', titlesize=titlesize)
    figure = plt.figure(figsize=figsize)
    i = 1
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # just plot the dataset first
    cm = plt.cm.RdBu
    cm_bright = cm


    # iterate over classifiers
    for name, clf in classifier_dict.items():
        y = clf.prediction_hard(X.to(device)).detach().cpu().numpy()
        remixed = False
        if gt_labels is not None:
            if len(set(y.tolist())) == len(set(gt_labels.tolist())):
                remixed = True
                D = max(y.max(), gt_labels.max()) + 1
                w = np.zeros((D, D), dtype=np.int64)
                for j in range(y.size):
                    w[y[j], gt_labels[j]] += 1
                # apply hungarian method to match colors
                ind = linear_sum_assignment(w.max() - w)
                ind = np.array(ind).transpose()
                ind_d = {i:j for i,j in ind}
                for j in range(y.shape[0]):
                    y[j] = ind_d[y[j]]


        ax = plt.subplot(1, len(classifier_dict) + 1, i)
        Z = clf.prediction_hard(torch.from_numpy(np.c_[xx.ravel(), yy.ravel()]).float().to(device)).detach().cpu().numpy()
        if remixed:
            for j in range(Z.shape[0]):
                Z[j] = ind_d[Z[j]]
        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        CS = ax.contourf(xx, yy, Z, cmap=cm, alpha=0.45)
        CS2 = ax.contour(CS, levels=CS.levels[::2], colors='black')

        # Plot the points
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cm_bright, edgecolors="black", s=s, linewidths=1)
        
        if plot_centers:
            try:
                # Plot Centers
                centers = []
                for label_i in set(y.tolist()):
                    center_i = np.mean(X.numpy()[y==label_i], axis=0).reshape(1,-1)
                    centers.append(center_i)
                centers = np.concatenate(centers, axis=0)
                ax.scatter(centers[:,0], centers[:,1], c="white", s=200, marker="X", edgecolors="k", linewidths=1)
            except Exception as e:
                print(e)
                pass

        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        if gt_labels is None:
            ax.set_title(name)        
        else:
            ari = adjusted_rand_score(y, gt_labels)
            ax.set_title(name + "-Clf. " + f" ARI:{ari:.2f}")        
        i += 1

    plt.tight_layout()
    if save is not None:
        plt.savefig(save, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.show()
    


def histogram(x):
    plt.hist(x)
    plt.show()