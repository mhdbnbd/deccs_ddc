import torch
from collections import OrderedDict
from utils.utils import predict_batchwise
from copy import deepcopy
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from model.autoencoder import FullyConnectedBlock
from utils.utils import squared_euclidean_distance
from collections import Counter


class MultinomialClassifier(torch.nn.Module):
    """MultinomialClassifier
    Args:
        layers: list of the different layer sizes from input_dim to number of classes
        batch_norm: bool, default=False, set True if you want to use torch.nn.BatchNorm1d
        dropout: float, default=None, set the amount of dropout you want to use.
        activation_fn: activation function from torch.nn, default=torch.nn.LeakyReLU, set the activation function for the hidden layers
        bias: bool, default=True, set False if you do not want to use a bias term in the linear layers
        output_fn: activation function from torch.nn, default=torch.nn.functional.softmax, set the activation function for the last layer
    Attributes:
        logit: torch.nn.Sequential, feed forward neural network to produce the embedding of the classifier
    """
    def __init__(self, layers, batch_norm=False, dropout=None, activation_fn=torch.nn.LeakyReLU, bias=True, output_fn=torch.nn.functional.softmax):
        super().__init__()
        self.layers = layers
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.activation_fn = activation_fn
        self.bias = bias
        self.output_fn = output_fn
        
        self.logit = FullyConnectedBlock(layers=self.layers, batch_norm=self.batch_norm, dropout=self.dropout, activation_fn=self.activation_fn, bias=self.bias, output_fn=None)


    def encode(self, x: torch.Tensor)->torch.Tensor:
        """
        Args:
            x: input data point, can also be a mini-batch of points
        
        Returns:
            embedded: the embedded data point with dimensionality embedding_size
        """
        return self.logit(x)
    
    def prediction_hard(self, x: torch.Tensor)->torch.Tensor:
        """
        Args:
            x: input data point, can also be a mini-batch of points
        
        Returns:
            prediction: predicted labels for x
        """
        y_soft = self.forward(x)
        return y_soft.argmax(1)
    
    def forward(self, x: torch.Tensor):
        """
        Args:
            x: input data point, can also be a mini-batch of points
        
        Returns:
            prediction: predicted probabilities for x
        """
        z = self.encode(x)
        soft_prediction = self.output_fn(z, dim=1)
        return soft_prediction

class _ClassifierEnsemble(torch.nn.Module):
    """Abstract Class for Ensemble of Classifiers
    Args:
        layers: list of the different layer sizes for the hidden layers. The dimensionality of the last layer is inferred from the cluster_label_dict
        cluster_label_dict: dict, labels predicted by each clustering algorithm
        eval_cluster_label_dict: dict,  labels predicted by each clustering algorithm used for validation
        classifier_type: torch.nn.Module, default=MultinomialClassifier, Class of Classifier to use
        batch_norm: bool, default=False, set True if you want to use torch.nn.BatchNorm1d
        dropout: float, default=None, set the amount of dropout you want to use.
        activation: activation function from torch.nn, default=torch.nn.LeakyReLU, set the activation function for the hidden layers
        bias: bool, default=True, set False if you do not want to use a bias term in the linear layers
        output_fn: activation function from torch.nn, default=torch.nn.functional.softmax, set the activation function for the last layer
        full_cluster_label_dict: dict, default=None, if not None will be used to initialize classifiers with correct n_clusters, because the train split can sometimes contain less than n_clusters, due to sampling issues.
        minimum_agreement: float, default=0.01, if weight_dict is passed than clustering results that have an agreement lower than minimum_agreement will be excluded.
    Attributes:
        classifiers: torch.nn.ModuleDict, dictionary of classifiers
        n_classifiers: int, number of classifiers

    """
    def __init__(self, layers, cluster_label_dict=None, classifier_type=MultinomialClassifier, batch_norm=False, dropout=None, activation_fn=torch.nn.LeakyReLU, bias=True, output_fn=torch.nn.functional.softmax, names=None, eval_cluster_label_dict=None, full_cluster_label_dict=None, minimum_agreement=1e-2):
        super().__init__()
        self.layers = layers
        self.classifier_type = classifier_type
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.activation_fn = activation_fn
        self.bias = bias
        self.output_fn = output_fn
        self.cluster_label_dict = cluster_label_dict
        self.minimum_agreement = minimum_agreement
        if self.cluster_label_dict is not None:
            self.cluster_label_dict = self._label_dict_to_tensors(self.cluster_label_dict)
            self.n_classifiers = len(self.cluster_label_dict)

        self.eval_cluster_label_dict = eval_cluster_label_dict
        if eval_cluster_label_dict is not None:
            self.eval_cluster_label_dict = self._label_dict_to_tensors(self.eval_cluster_label_dict)
        
        self.full_cluster_label_dict = full_cluster_label_dict
        if full_cluster_label_dict is not None:
            self.full_cluster_label_dict = self._label_dict_to_tensors(self.full_cluster_label_dict)
            
    
    def to(self, device):
        for name_i, c_i in self.classifiers.items():
            c_i.to(device)
            self.cluster_label_dict[name_i].to(device)
            if self.eval_cluster_label_dict is not None:
                self.eval_cluster_label_dict[name_i].to(device)
        return self

    def _label_dict_to_tensors(self, d):
        new_d = OrderedDict()
        for name, labels in d.items():
            if isinstance(labels, torch.Tensor):
                new_d[name] = labels.long()
            else:
                new_d[name] = torch.from_numpy(labels).long()
        return new_d

    def _init_classifiers(self):
        raise NotImplementedError()


def feed_one_batch_classifier_ensemble(classifier_ensemble, x: torch.Tensor, indices: torch.Tensor, weight_dict=None, loss_fn=torch.nn.CrossEntropyLoss()):
    """
    Args:
        classifier_ensemble: torch.nn.Module, ensemble of classifiers to use
        x: input data point, can also be a mini-batch of points
        indices: indices of cluster labels to be used
        weight_dict: dict, default=None, weight for each classifier loss
        loss_fn: torch.nn, default=torch.nn.CrossEntropyLoss, classification loss
    Returns:
        loss: summed loss over all classifiers, which should be passed to the optimizer
        loss_dict: dictionary of losses of each classifier. Returned for debugging purposes
        pred_score_dict: dictionary of detached pred_scores aka logits of each clasifier
    """
    loss_dict = OrderedDict({})
    pred_score_dict = OrderedDict({})
    loss = 0
    for name_i, c_i in classifier_ensemble.classifiers.items():
        if c_i.training:
            s_labels = classifier_ensemble.cluster_label_dict[name_i][indices]
        else:
            s_labels = classifier_ensemble.eval_cluster_label_dict[name_i][indices]
        pred_score = c_i.encode(x)
        loss_i = loss_fn(pred_score, s_labels.to(pred_score.device))
               
        if weight_dict is None:
            loss += loss_i
        else:
            if weight_dict[name_i] > classifier_ensemble.minimum_agreement:
                loss += weight_dict[name_i] * loss_i
        loss_dict[name_i] = loss_i.item()
        pred_score_dict[name_i] = pred_score.detach()
    return loss, loss_dict, pred_score_dict

class ClassifierEnsemble(_ClassifierEnsemble):
    """Wrapper Class for Ensemble of Classifiers
    Args:
        layers: list of the different layer sizes for the hidden layers. The dimensionality of the last layer is inferred from the cluster_label_dict
        cluster_label_dict: dict, labels predicted by each clustering algorithm
        eval_cluster_label_dict: dict,  labels predicted by each clustering algorithm used for validation
        classifier_type: torch.nn.Module, default=MultinomialClassifier, Class of Classifier to use
        batch_norm: bool, default=False, set True if you want to use torch.nn.BatchNorm1d
        dropout: float, default=None, set the amount of dropout you want to use.
        activation: activation function from torch.nn, default=torch.nn.LeakyReLU, set the activation function for the hidden layers
        bias: bool, default=True, set False if you do not want to use a bias term in the linear layers
        output_fn: activation function from torch.nn, default=torch.nn.functional.softmax, set the activation function for the last layer
    Attributes:
        classifiers: torch.nn.ModuleDict, dictionary of classifiers
        n_classifiers: int, number of classifiers

    """
    def __init__(self, **kwargs):
        kwargs["classifier_type"] = MultinomialClassifier
        super().__init__(**kwargs)
        if self.cluster_label_dict is not None:
            # init classifiers    
            self.classifiers = self._init_classifiers()

    
    def _init_classifiers(self):
        classifiers_dict = OrderedDict()
        for name_i, labels in self.cluster_label_dict.items():
            if self.full_cluster_label_dict is not None:
                output_dim = len(set(self.full_cluster_label_dict[name_i].tolist()))
            else:
                output_dim = len(set(labels.tolist()))
            classifiers_dict[name_i] = self.classifier_type(layers=self.layers+[output_dim], batch_norm=self.batch_norm, dropout=self.dropout, activation_fn=self.activation_fn, bias=self.bias, output_fn=self.output_fn)
        return torch.nn.ModuleDict(classifiers_dict)

    def forward(self, x: torch.Tensor, indices: torch.Tensor, weight_dict=None, loss_fn=torch.nn.CrossEntropyLoss()):
        """
        Args:
            x: input data point, can also be a mini-batch of points
            indices: indices of cluster labels to be used
            weight_dict: dict, default=None, weight for each classifier loss
            loss_fn: torch.nn, default=torch.nn.CrossEntropyLoss, classification loss
        Returns:
            loss: summed loss over all classifiers, which should be passed to the optimizer
            loss_dict: dictionary of losses of each classifier. Returned for debugging purposes
            pred_score_dict: dictionary of detached pred_scores aka logits of each clasifier
        """
        loss, loss_dict, pred_score_dict = feed_one_batch_classifier_ensemble(classifier_ensemble=self, x=x, indices=indices, weight_dict=weight_dict, loss_fn=loss_fn)
        return loss, loss_dict, pred_score_dict
