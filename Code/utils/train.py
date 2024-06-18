# internal packages
import os
from itertools import cycle
from collections import OrderedDict
import pickle
# external packages
import numpy as np
import torch
# own packages
from utils.utils import predict_batchwise
from utils.utils import encode_batchwise_from_data
from utils.utils import squared_euclidean_distance
from utils.utils import int_to_one_hot
from utils.data import Dataset_with_indices, shuffle_dataset
from utils.ensemble import calculate_weights
from utils.ensemble import calculate_max_normalized_weights 
from utils.ensemble import get_classifier_uncertainty_weights
from utils.ensemble import get_prediction_dict
from utils.ensemble import get_assignment_matrix_dict
from utils.ensemble import calculate_avg_agreement
from utils.ensemble import create_cluster_dict
from utils.ensemble import cluster_dict_train_test_split
from utils.visualize import plot_clusterings
from utils.visualize import plot_classifier_probabilities
from utils.ensemble import remove_small_clusters

class LossNanError(Exception):
    def __init__(self, message):            
        super().__init__(message)

class EarlyStopping():
    # Adapted from https://debuggercafe.com/using-learning-rate-scheduler-and-early-stopping-with-pytorch/
    """Early stopping to stop the training when the loss does not improve after
    certain epochs.
    Args:
        patience: int, default=10, how many epochs to wait before stopping when loss is not improving
        min_delta: float, default=1e-4, minimum difference between new loss and old loss for new loss to be considered as an improvement
        verbose: bool, default=False, if True will print INFO statements
    Attributes:
        counter: integer counting the consecutive epochs without improvement
        best_loss: best loss achieved before stopping
        early_stop: boolean indicating whether to stop training or not
    """
    def __init__(self, patience=10, min_delta=1e-4, verbose=False):

        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose

        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

def calculate_means_with_one_hot_assigment_matrix(x, assignments):
    label_sums = (x.unsqueeze(1) * assignments.unsqueeze(2)).sum(0)
    frequency_per_cluster =  assignments.sum(0).unsqueeze(1)
    label_means = label_sums / frequency_per_cluster
    return label_means.detach()

def calculate_means_with_one_hot_assigment_matrix_and_past_means(x, assignments, past_means, weight=0.5, fp_threshold=1e-12):
    label_sums = (x.unsqueeze(1) * assignments.unsqueeze(2)).sum(0)
    frequency_per_cluster =  assignments.sum(0).unsqueeze(1)
    nonzero_mask = (frequency_per_cluster.squeeze(1) > fp_threshold)
    label_means =  label_sums[nonzero_mask] / frequency_per_cluster[nonzero_mask]
    past_means[nonzero_mask] = past_means[nonzero_mask] * weight + label_means * (1-weight)
    return past_means.detach()

def calculate_regularizer_losses(original, embedded, model, regularizer_list, u_original=None, u_embedded=None):
    """Utility function for calculating the regularizer losses"""
    loss = 0
    loss_dict = OrderedDict()
    for regularizer_i in regularizer_list:
        if regularizer_i.__class__.__name__ == ReconstructionLoss.__name__:
            rec_loss = regularizer_i(original=original, embedded=embedded, model=model, u_original=u_original, u_embedded=u_embedded)
            loss += rec_loss
            loss_dict[regularizer_i.__class__.__name__] = rec_loss.item()
    return loss, loss_dict
    

class ReconstructionLoss(torch.nn.Module):
    """ReconstructionLoss
    Args:
        weight: float, default=1.0, weighting term for the reconstruction loss
        loss_fn: torch.nn, default=torch.nn.MSELoss, loss function to be used.
    """

    def __init__(self, weight=1.0, loss_fn=torch.nn.MSELoss):
        super(ReconstructionLoss, self).__init__()
        self.loss_fn = loss_fn()
        self.weight = weight

    def forward(self, original, embedded, model, u_original=None, u_embedded=None):
        """Calculates the reconstruction error
        Args:
            original: torch.tensor, original data point
            embedded: torch.tensor, embedded data point to be decoded. 
            model: torch.nn.Module, autoencoder to be used
        """
        reconstructed = model.decode(embedded)
        rec_loss =  self.loss_fn(reconstructed, original)
        if (u_original is not None) and (u_embedded is not None):
            u_reconstructed = model.decode(u_embedded)
            rec_loss +=  self.loss_fn(u_reconstructed, u_original)
        return self.weight * rec_loss


class MeanLoss(torch.nn.Module):
    """MeanLoss
    Args:
        soft_assignments: bool, default=True, if set True will calculate the means of each class and their assignments using soft assignments based on the classifier prediction. 
                      If False the centers will be computed on the hard predicted labels
        assignment_matrix_dict: dict, default=None, dictionary of one hot encoded cluster labels per clustering algorithm.
        weight: float, default=1.0, optional weight for the summed mean loss.
        minimum_agreement: float, default=0.01, if weight_dict is passed than clustering results that have an agreement lower than minimum_agreement will be excluded.
        use_clf: bool, default=True, specify if classifiers should be used in computation
    Attributes:
        label_means_dict: dict, last label_means that where calculated
    """
    def __init__(self, soft_assignments=True, assignment_matrix_dict=None, assignment_matrix_eval_dict=None, weight=1.0, minimum_agreement=1e-2, use_clf=True):
        super(MeanLoss, self).__init__()
        self.soft_assignments = soft_assignments
        self.assignment_matrix_dict = assignment_matrix_dict
        self.assignment_matrix_eval_dict = assignment_matrix_eval_dict
        self.weight = weight
        self.minimum_agreement = minimum_agreement
        self.use_clf = use_clf
        self.label_means_dict = None
        self.loss = self.loss_hard

    def init_label_means_dict(self, embedded, assignment_matrix_dict=None):
        if assignment_matrix_dict is not None:
            self.assignment_matrix_dict = assignment_matrix_dict
        if self.assignment_matrix_dict is None:
            raise ValueError("If self.assignment_matrix_dict is None then assignment_matrix_dict cannot be None")
        
        self.label_means_dict = OrderedDict({name:None for name,_ in self.assignment_matrix_dict.items()}) 
        for name_i, assignments_i in self.assignment_matrix_dict.items():
            _, label_means_i = self.loss(embedded=embedded, assignments=assignments_i, past_means=None)
            self.label_means_dict[name_i] = label_means_i
            print(name_i, label_means_i.shape)

    def loss_hard(self, embedded, assignments, past_means=None, soft_assignments=None, u_embedded=None, u_assignments=None, u_soft_assignments=None):
        if past_means is None:
            label_means = calculate_means_with_one_hot_assigment_matrix(x=embedded, assignments=assignments)
        else:
            label_means = calculate_means_with_one_hot_assigment_matrix_and_past_means(x=embedded, assignments=assignments, past_means=past_means)
        dist = squared_euclidean_distance(label_means, embedded)
        if soft_assignments is None:
            soft_assignments = 1
        # by multiplying with assignments we only move data points with the same assignment to their center
        loss =  (dist * assignments * soft_assignments).sum(1).mean()
        if u_embedded is not None:
            if u_assignments is not None and u_soft_assignments is not None:
                dist = squared_euclidean_distance(label_means, u_embedded)
                loss +=  (dist * u_assignments * u_soft_assignments).sum(1).mean()
        return loss, label_means

    def forward(self, embedded, classifier_dict, pred_score_dict=None, assignment_indices=None, weight_dict=None, u_embedded=None):
        """
        Args:
            embedded: embedded input data point, can also be a mini-batch of points
            classifier_dict: dict, dictionary of classifiers to be used for calculating the mean loss
            pred_score_dict: dict, default=None, logit predictions of classifiers, 
                             if not specified predictions will be calculated again
                             and creating some overhead.
            assignment_indices: torch.tensor, default=None, indices of current data points in embedded. 
                               Is used to access elements in assignment_matrix_dict and 
                               constructed via ensemble.get_assignment_matrix_dict.
            weight_dict: dict, default=None, weights for loss per clustering algorithm
        Returns:
            loss: summed loss over all classifiers, which should be passed to the optimizer
            loss_dict: dictionary of losses of each classifier. Returned for debugging purposes
        """
        loss_dict = OrderedDict({})
        loss = 0
        soft_assignments = None
        u_assignments = None
        u_soft_assignments = None
        if self.label_means_dict is None:
            self.label_means_dict = OrderedDict({name:None for name,_ in classifier_dict.items()}) 
        for name_i, c_i in classifier_dict.items():

            if self.use_clf:
                if pred_score_dict is None:
                    pred_score = c_i.encode(embedded).detach()
                else:
                    # can be used to save computation time
                    pred_score = pred_score_dict[name_i].detach()
                
                soft_assignments = torch.nn.functional.softmax(pred_score, dim=1).detach()
                if self.soft_assignments:
                    assignments_i = soft_assignments

            if not self.soft_assignments:
                if self.assignment_matrix_dict is None:
                    # predictions are dependent on classifier so they might change
                    assignments_i = assignments_i.argmax(1)
                    assignments_i = int_to_one_hot(assignments_i, pred_score.shape[1])
                elif assignment_indices is not None:
                    #predictions are fixed to cluster labels
                    if self.training:
                        assignments_i = self.assignment_matrix_dict[name_i][assignment_indices]
                    else:
                        assignments_i = self.assignment_matrix_eval_dict[name_i][assignment_indices]
                else:
                    raise ValueError("assignment_indices cannot be None if self.assignment_matrix_dict is not None")
            if u_embedded is not None:
                if self.use_clf:
                    u_pred_score = c_i.encode(u_embedded).detach()
                    # divide by small number to get hard_assignments
                    u_assignments = torch.nn.functional.softmax(u_pred_score/1e-4, dim=1).detach()
                    u_soft_assignments = torch.nn.functional.softmax(u_pred_score, dim=1).detach()

            loss_i, label_means_i = self.loss(embedded=embedded, assignments=assignments_i, past_means=self.label_means_dict[name_i], soft_assignments=soft_assignments, u_embedded=u_embedded, u_assignments=u_assignments, u_soft_assignments=u_soft_assignments)
                
            if self.training:
                self.label_means_dict[name_i] = label_means_i.detach()
            if weight_dict is None:
                loss += loss_i
            else:
                if weight_dict[name_i] > self.minimum_agreement:
                    loss += weight_dict[name_i] * loss_i
            loss_dict[name_i] = loss_i.item()
        return self.weight*loss, loss_dict


def sigmoid_rampup(current, rampup_length):
    # from https://github.com/vikasverma1077/ICT/blob/03ebd7ed8b384507f89b3fff321a5489523e5af7/mean_teacher/ramps.py#L19
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))

def get_current_consistency_weight(global_step, number_of_batches_in_dataloader, unsup_mixup_consistency_weight, consistency_rampup_ends, consistency_rampup_starts, epoch):
    # from https://github.com/vikasverma1077/ICT/blob/03ebd7ed8b384507f89b3fff321a5489523e5af7/main.py#L830
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    epoch = epoch - consistency_rampup_starts
    epoch = epoch + global_step / number_of_batches_in_dataloader

    return unsup_mixup_consistency_weight * sigmoid_rampup(epoch, consistency_rampup_ends - consistency_rampup_starts)

def validate_autoencoder(model, dataloader, loss_fn, device=torch.device("cpu")):
    with torch.no_grad():
        model.eval()
        loss = 0
        for batch in dataloader:
            batch = batch[0].to(device)
            reconstruction = model(batch)
            loss += loss_fn(reconstruction, batch)
        loss /= len(dataloader)
    return loss


def fit_ae(model, training_iterations, lr, batch_size=128, data=None, data_eval=None, dataloader=None, evalloader=None, optimizer_fn=torch.optim.Adam, loss_fn=torch.nn.MSELoss, patience=10, device=torch.device("cpu"), model_path=None, print_step=2500):
    """ Function to train the autoencoder
    Args:
        model: torch.nn.Module, autoencoder model to be trained
        training_iterations: int, number of training iterations for training
        lr: float, learning rate to be used for the optimizer_fn
        batch_size: int, default=128
        data: np.ndarray, default=None, train data set
        data_eval: np.ndarray, default=None, evaluation data set
        dataloader: torch.utils.data.DataLoader, default=None, dataloader to be used for training
        evalloader: torch.utils.data.DataLoader, default=None, dataloader to be used for validation
        optimizer_fn: torch.optim, default=torch.optim.Adam, optimizer to be used
        loss_fn: torch.nn, default=torch.nn.MSELoss, loss function to be used for reconstruction
        patience: int, default=10, patience parameter for learning rate scheduler and early stopping
        device: torch.device, default=torch.device('cpu'), device to be trained on
        model_path: str, default=None, if specified will save the trained model to the location
        print_step: int, default=2500, specifies how often the losses are printed
    raises:
        ValueError: data cannot be None if dataloader is None
    """
    if dataloader is None:
        if data is None:
            raise ValueError("data must be specified if dataloader is None")
        dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.from_numpy(data).float()),
                                                 batch_size=batch_size,
                                                 shuffle=True)
    if evalloader is None:
        if data_eval is not None:
            evalloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.from_numpy(data_eval).float()),
                                                    batch_size=batch_size,
                                                    shuffle=False)
    
    params_dict = {'params': model.parameters(), 'lr': lr}
    optimizer = optimizer_fn(**params_dict)
    
    early_stopping = EarlyStopping(patience=patience)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=patience//2, verbose=True)
    best_loss = np.inf
    i = 0
    # training loop
    while(i < training_iterations):
        model.train()
        for batch in dataloader:
            batch = batch[0].to(device)
            reconstruction = model(batch)
            loss = loss_fn(reconstruction, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i > training_iterations:
                print("Stop training")
                break
            if i % 1000 == 0 or i == (training_iterations - 1):
                print(f"Iteration {i}/{training_iterations} - Reconstruction loss: {loss.item():.6f}")
            i += 1
        
        if evalloader is not None:
            val_loss = validate_autoencoder(model=model, dataloader=evalloader, loss_fn=loss_fn, device=device)
            if (i-1) % print_step == 0 or i == training_iterations:
                print(f"Iteration {i-1} EVAL loss total: {val_loss.item():.6f}")
            early_stopping(val_loss)
            if val_loss < best_loss:
                best_loss = val_loss
                best_iteration = i
                if model_path is not None:
                    torch.save({"model":model.state_dict(), "best_iteration": best_iteration, "best_loss": best_loss}, model_path)

            if early_stopping.early_stop:
                print(f"Stop training at iteration {i-1}")
                print(f"Best Loss: {best_loss:.6f}, Last Loss: {val_loss:.6f}")
                break
            scheduler.step(val_loss)
    model.eval()
    if evalloader is None:
        if model_path is not None:
            torch.save(model.state_dict(), model_path)

def validate_classifiers(classifier_ensemble, dataloader, loss_fn, unlabelledloader=None, device=torch.device("cpu")):
    with torch.no_grad():
        classifier_ensemble.eval()
        loss = 0
        loss_dict = None
        for s_batch, indices in dataloader:
            s_batch = s_batch.to(device)
            indices = indices.to(device)
            
            loss_i, loss_dict_i, _ = classifier_ensemble(s_batch, indices, weight_dict=None, loss_fn=loss_fn)
            loss += loss_i
            if loss_dict is None:
                loss_dict = loss_dict_i
            else:
                for name, loss_j in loss_dict_i.items():
                    loss_dict[name] += loss_j
        loss /= len(dataloader)
        for name, _ in loss_dict_i.items():
            loss_dict[name]  /= len(dataloader)
    return loss, loss_dict



def train_classifiers(classifier_ensemble, training_iterations, lr, data, unlabelled_data=None, data_eval=None, batch_size=128, optimizer_fn=torch.optim.Adam, classifier_loss_fn=torch.nn.CrossEntropyLoss, patience=10, device=torch.device("cpu"), print_step=2500, model_path=None):
    """ Function to train all classifiers in parallel.
    Args:
        classifier_ensemble: ClassifierEnsemble, ensemble of model.ensemble.MultinomialClassifier, initialised classifiers to be trained
        training_iterations: int, number of training iterations for training
        lr: float, learning rate to be used for the optimizer_fn
        data: np.ndarray, data set for training
        unlabelled_data, np.ndarray, default=None, data set for training without cluster labels
        data_eval: np.ndarray, default=None, data set for validation
        batch_size: int, default=128
        optimizer_fn: torch.optim, default=torch.optim.Adam, optimizer to be used
        classifier_loss_fn: torch.nn, default=torch.nn.CrossEntropyLoss, loss function to be used for classifier
        patience: int, default=10, patience parameter for learning rate scheduler and early stopping
        device: torch.device, default=torch.device('cpu'), device to be trained on
        model_path: str, default=None, if specified will save the trained model to the location
        print_step: int, default=2500, specifies how often the losses are printed
    """
    
    dataloader = torch.utils.data.DataLoader(Dataset_with_indices(torch.from_numpy(data).float(),),
                                                batch_size=batch_size,
                                                shuffle=True)
    if unlabelled_data is not None:
        unlabelledloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.from_numpy(unlabelled_data).float(),),
                                            batch_size=batch_size,
                                            shuffle=True)
    else:
        unlabelledloader = None

    if data_eval is not None:
        evalloader = torch.utils.data.DataLoader(Dataset_with_indices(torch.from_numpy(data_eval).float(),),
                                                    batch_size=batch_size,
                                                    shuffle=False)

    classifier_loss = classifier_loss_fn()

    classifier_ensemble.to(device)
    optimizer = optimizer_fn(classifier_ensemble.parameters(), lr)
    early_stopping = EarlyStopping(patience=patience)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.9, patience=patience//2, verbose=True)
    
    best_loss = np.inf
    i = 0
    epoch_i = 0
    while(i < training_iterations):
        print_eval = False
        classifier_ensemble.train()
        if unlabelledloader is not None:
            sampler = zip(cycle(dataloader), unlabelledloader)
        else:
            sampler = zip(dataloader) 
        for batch_data in sampler:
            s_batch = batch_data[0][0].to(device)
            indices = batch_data[0][1].to(device)
            if unlabelledloader is not None:
                u_batch = batch_data[1][0].to(device)
            loss, loss_dict, _ = classifier_ensemble(s_batch, indices, weight_dict=None, loss_fn=classifier_loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % print_step == 0 or i == (training_iterations-1):
                print(f"Epoch {epoch_i} - iteration {i} MB loss total: {loss.item():.6f}")
                for name, loss_i in loss_dict.items():
                    print(f"\tClassifier {name} loss: {loss_i:.6f}")
                print_eval = True
            if i == training_iterations: break
            i += 1
        epoch_i += 1
        if data_eval is not None:
            val_loss, val_loss_dict = validate_classifiers(classifier_ensemble=classifier_ensemble,
                                                           dataloader=evalloader,
                                                           unlabelledloader=unlabelledloader,
                                                           loss_fn=classifier_loss, device=device)
            early_stopping(val_loss)
            if val_loss < best_loss:
                best_iteration = i
                best_loss = val_loss
                if model_path is not None:
                    if classifier_ensemble.full_cluster_label_dict is not None:
                        cluster_label_dict = classifier_ensemble.full_cluster_label_dict
                    else:
                        cluster_label_dict = classifier_ensemble.cluster_label_dict
                    torch.save({"classifiers":classifier_ensemble.state_dict(), "cluster_dict":cluster_label_dict,  "best_iteration": best_iteration, "best_loss": best_loss}, model_path)
            if print_eval:
                print(f"Epoch {epoch_i} - iteration {i-1} EVAL loss total: {val_loss.item():.6f}")
                for name, loss_i in val_loss_dict.items():
                    print(f"\tClassifier {name} EVAL loss: {loss_i:.6f} (Current Best: {best_loss:.6f})")
            if early_stopping.early_stop:
                print(f"Stop training at iteration {i-1}")
                print(f"Early Stopping Best Loss: {early_stopping.best_loss:.4f}, Last Loss: {val_loss:.4f}")
                break
            scheduler.step(val_loss)
            
    if model_path is not None:
        # load best classifier
        sd = torch.load(model_path)
        classifier_ensemble.load_state_dict(sd["classifiers"])
    classifier_ensemble.eval()

def validate_representation(classifier_ensemble, dataloader, model, classifier_loss, weight_dict=None, mean_loss=None, regularizer_list=None, unlabelledloader=None, use_clf=True, device=torch.device("cpu")):
    with torch.no_grad():
        model.eval()
        classifier_ensemble.eval()
        if mean_loss is not None:
            mean_loss.eval()
        loss = 0
        u_batch = None
        u_embedded = None
        pred_score_dict = None
        for s_batch, indices in dataloader:
            s_batch = s_batch.to(device)
            indices = indices.to(device)
            
            loss_i = 0
            embedded = model.encode(s_batch)

            if use_clf:
                classification_loss, loss_dict, pred_score_dict = classifier_ensemble(embedded, indices, weight_dict=weight_dict, loss_fn=classifier_loss)
                loss_i += classification_loss
            
            if mean_loss is not None:
                mean_loss_sum, _ = mean_loss(embedded=embedded, classifier_dict=classifier_ensemble.classifiers, pred_score_dict=pred_score_dict, assignment_indices=indices, weight_dict=weight_dict, u_embedded=u_embedded)
                loss_i += mean_loss_sum
            
            if regularizer_list is not None:
                reg_loss_sum, reg_loss_dict = calculate_regularizer_losses(original=s_batch, embedded=embedded, model=model, regularizer_list=regularizer_list, u_original=u_batch, u_embedded=u_embedded)
                loss_i += reg_loss_sum
            loss += loss_i
        
        loss /= len(dataloader)
    return loss

def train_representation(classifier_ensemble, model, training_iterations, lr, data, unlabelled_data=None, data_eval=None, batch_size=128, optimizer_fn=torch.optim.Adam, use_clf=True, fix_classifiers=True, classifier_loss_fn=torch.nn.CrossEntropyLoss, mean_loss=None, regularizer_list=None, weight_dict=None, patience=10, device=torch.device("cpu"), print_step=2500, model_path=None):
    """ Function to update the representation. Does not make use of unlabelled data yet.
    Args:
        classifier_ensemble: ClassifierEnsemble, ensemble of model.ensemble.MultinomialClassifier
        model: torch.nn.Module, specifiy model to update representation
        training_iterations: int, number of training iterations for training
        lr: float, learning rate to be used for the optimizer_fn
        data: np.ndarray, data set for training
        unlabelled_data, np.ndarray, default=None, data set for training without cluster labels
        data_eval: np.ndarray, default=None, data set for validation
        batch_size: int, default=128
        optimizer_fn: torch.optim, default=torch.optim.Adam, optimizer to be used
        fix_classifiers: bool, default=True, set to True if you want to keep classifiers fixed
        classifier_loss_fn: torch.nn, default=torch.nn.CrossEntropyLoss, loss function to be used for classifier
        mean_loss, MeanLoss, default=None, calculates the mean loss to pull embedded data points in the same cluster closer together
        regularizer_list: dict, default=None, regularizers to be used for the training
        weight_dict: dict, default=None, weight for each classifier loss
        patience: int, default=10, patience parameter for learning rate scheduler and early stopping
        device: torch.device, default=torch.device('cpu'), device to be trained on
        model_path: str, default=None, if specified will save the trained model to the location
        print_step: int, default=2500, specifies how often the losses are printed
    raises:
        ValueError: data cannot be None if dataloader is None
    """
    dataset = Dataset_with_indices(torch.from_numpy(data).float(),)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             # Because drop last is false we need to make sure that the batch_size is not larger than the data size
                                             batch_size=data.shape[0] if data.shape[0] < batch_size else batch_size,
                                             shuffle=True,
                                             drop_last=True)
    if unlabelled_data is not None:
        unlabelledloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.from_numpy(unlabelled_data).float(),),
                                            batch_size=batch_size,
                                            shuffle=True)
    else:
        unlabelledloader = None

    if data_eval is not None:
        evalloader = torch.utils.data.DataLoader(Dataset_with_indices(torch.from_numpy(data_eval).float(),),
                                                    batch_size=batch_size,
                                                    shuffle=False)


    classifier_loss = classifier_loss_fn()
    if fix_classifiers:
        params = model.parameters()
    else:
        params = list(classifier_ensemble.parameters()) + list(model.parameters())
    optimizer = optimizer_fn(params, lr)
    
    early_stopping = EarlyStopping(patience=patience)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.9, patience=patience//2, verbose=True)

    model.to(device)
    classifier_ensemble.to(device)

    best_loss = np.inf
    i = 0
    epoch_i = 0

    if weight_dict is not None:
        avg_agreement = calculate_avg_agreement(weight_dict)

    u_batch = None
    u_embedded = None
    loss_dict = None
    pred_score_dict = None
    while(i < training_iterations):
        print_eval = False
        model.train()
        classifier_ensemble.train()
        if mean_loss is not None:
            mean_loss.train()

        if unlabelledloader is not None:
            sampler = zip(cycle(dataloader), unlabelledloader)
        else:
            sampler = zip(dataloader) 
        for batch_data in sampler:
            s_batch = batch_data[0][0].to(device)
            indices = batch_data[0][1].to(device)
            if unlabelledloader is not None:
                u_batch = batch_data[1][0].to(device)
                u_embedded = model.encode(u_batch)
            loss = 0
            embedded = model.encode(s_batch)
            if use_clf:
                classification_loss, loss_dict, pred_score_dict = classifier_ensemble(x=embedded, indices=indices, weight_dict=weight_dict, loss_fn=classifier_loss)
                loss += classification_loss
            
            if mean_loss is not None:
                mean_loss_sum, mean_loss_dict = mean_loss(embedded=embedded, classifier_dict=classifier_ensemble.classifiers, pred_score_dict=pred_score_dict, assignment_indices=indices, weight_dict=weight_dict, u_embedded=u_embedded)
                loss += mean_loss_sum
                
            
            # Divide by number of classifiers
            # reg_loss_sum does not depend on the size of the classifier_ensemble and thus it is not needed to scale it with the number of classifiers
            loss /= classifier_ensemble.n_classifiers
            
            if regularizer_list is not None:
                reg_loss_sum, reg_loss_dict = calculate_regularizer_losses(original=s_batch, embedded=embedded, model=model, regularizer_list=regularizer_list, u_original=u_batch, u_embedded=u_embedded)
                if weight_dict is not None:
                    # if agreement is low we weight the regularizer losses higher
                    reg_loss_sum *= (1 - avg_agreement) 
                loss += reg_loss_sum

            if torch.isnan(loss):
                if use_clf:
                    print("classification_loss: ", classification_loss)
                if mean_loss is not None:
                    print("mean_loss_sum: ", mean_loss_sum) 
                print("reg_loss_sum: ", reg_loss_sum) 
                raise LossNanError(f"unexpected value for loss={loss}, check data for nans and/or reduce learning rate")


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % print_step == 0 or i == (training_iterations-1):
                print(f"Epoch {epoch_i} - iteration {i} MB loss total: {loss.item():.6f}")
                for name, _ in classifier_ensemble.classifiers.items():
                    if loss_dict is not None:
                        print(f"\t{name} classification loss: {loss_dict[name]:.6f}")
                    if mean_loss is not None:
                        print(f"\t{name} mean loss: {mean_loss_dict[name]:.6f}")
                if regularizer_list is not None:
                    for name, loss_i in reg_loss_dict.items():
                        print(f"\t{name}: {loss_i:.6f}")
                print_eval = True
                
            if i == training_iterations: break
            i += 1
        epoch_i += 1
        if data_eval is not None:
            val_loss = validate_representation(classifier_ensemble=classifier_ensemble,
                                               dataloader=evalloader,
                                               unlabelledloader=unlabelledloader,
                                               model=model,
                                               classifier_loss=classifier_loss,
                                               mean_loss=mean_loss,
                                               weight_dict=weight_dict,
                                               regularizer_list=regularizer_list,
                                               use_clf=use_clf,
                                               device=device)
            if print_eval:
                print(f"Epoch {epoch_i} - iteration {i-1} EVAL loss total: {val_loss.item():.6f} (Current Best: {best_loss:.6f})")
            early_stopping(val_loss)
            if val_loss < best_loss:
                best_iteration = i
                best_loss = val_loss
                if model_path is not None:
                    if classifier_ensemble.full_cluster_label_dict is not None:
                        cluster_label_dict = classifier_ensemble.full_cluster_label_dict
                    else:
                        cluster_label_dict = classifier_ensemble.cluster_label_dict
                    torch.save({"model":model.state_dict(), "classifiers":classifier_ensemble.state_dict(), "cluster_dict":cluster_label_dict,  "best_iteration": best_iteration, "best_loss": best_loss}, model_path)

            if early_stopping.early_stop:
                print(f"Stop training at iteration {i-1}")
                print(f"Early Stopping Best Loss: {early_stopping.best_loss:.4f}, Last Loss: {val_loss:.4f}")
                break
            scheduler.step(val_loss)
        

    if model_path is not None:
        # load best model
        sd = torch.load(model_path)
        model.load_state_dict(sd["model"])
        classifier_ensemble.load_state_dict(sd["classifiers"])
    model.eval()
    classifier_ensemble.eval()
    return model, classifier_ensemble, early_stopping.best_loss

def alternate_optimization(model, data, gt_labels, clusterer_dict_fn, max_rounds, max_iterations, clf_iterations, clf_lr, repr_lr, clf_optimizer_fn, repr_optimizer_fn, ensemble_class, use_unlabelled=False, regularizer_weight=1, clf_layers=None,  fix_classifiers=False, train_split_ratio=0.5, subsample_size=400, stability_threshold=0.001, repr_lr_decrease_rate=0.9, use_agreement_weighting=True, patience=10, use_clf=True, use_rec=True, use_mean=True, mean_weight_rampup=None, batch_size=32, n_clusters=None, pretrain_classifiers=True, device=torch.device("cpu"), model_path=None, save_plots=None, print_step=2500, hardening_agreement_threshold=1.0, patience_rounds=3):
    """ Function to update the classifiers and representation in an alternating manner.
    Args:
        model: torch.nn.Module, specifiy model to update representation
        data: np.ndarray, training data set
        gt_labels: np.ndarray, ground truth training cluster labels
        clusterer_dict_fn: function, function that returns a dictionary with different parameterised clustering algorithms to be used
        max_rounds: int, maximum number alternating training rounds. One round is equal to one update of the classifiers and the representation. 
        max_iterations: int, maximum number of training iterations for classifiers and representation. 
        clf_lr: float, learning rate to be used for the classifier optimizer clf_optimizer_fn
        repr_lr: float, learning rate to be used for the representation learning repr_optimizer_fn
        clf_optimizer_fn: torch.optim, optimizer to be used
        repr_optimizer_fn: torch.optim, optimizer to be used
        ensemble_class: ClassifierEnsemble, include which classifier ensemble you want to use
        use_unlabelled: bool, default=False, set to True if you want to use unlabelled data during training. 
        regularizer_weight: float, default = 1 for rec.
        clf_layers: list of ints, default=None, if None the classifiers will be linear. Specify layers for the classifier, 
                    e.g. [embedding_size, 20, 20], the output layer is determined automatically based on the number of classes.
        fix_classifiers: bool, default=False, specify whether you want to fix classifiers during the representation update
        train_split_ratio: float, ratio of training and validation data from clustering results. 
                           Calculated as train_split_ratio*subsample_size, e.g. 0.5*400 = 200 datapoints used for training.
        subsample_size: int, default=400, number of data points that should be used for clustering
        stability_threshold: float, default=0.001, stopping criterium if stable agreement over subsequent rounds is reached. Very small values will cause longer training and in the worst case until max_rounds are reached.
        repr_lr_decrease_rate: float, default=0.9, decrease rate of repr_lr
        use_agreement_weighting: bool, default=True, set True if losses should be weighted by agreement
        use_clf: bool, default=True, if True then classifiers are included in the objective and used to constraint the embedded space
        use_rec: bool, default=True, if True then reconstruction loss is used to regularize the embedded space
        use_mean: bool, default=True, if True then embedded data points will be moved cluster to their cluster prototype represented as the label mean.
        mean_weight_rampup: float>0, default=None, if set to a value > 0, it will rampup a weighting parameter over the course of training until half of max_rounds and then the mean_weight_rampup value is reached.
        patience: int, default=10, patience parameter for learning rate scheduler and early stopping
        batch_size: int, default=32
        n_clusters: int, default=None, number of clusters to use for ensemble. If None will be set to number of ground truth clusters based on gt_labels.
        pretrain_classifiers: bool, default=False, determine whether classifiers should be pretrained
        device: torch.device, default=torch.device('cpu'), device to be trained on
        model_path: str, default=None, if specified will save the trained model to the location
        save_plots: str, default=None, if specified will save the generated plots to the locations
        print_step: int, default=2500, specifies how often the losses are printed
        hardening_agreement_threshold: float, default=1.0, value should be between 0 and 1. If the value is 1 or larger it will be ignored. The threshold indicates when the agreement procedure should harden, i.e., get closer to full agreement.
                                       By setting to 1, we are currently not using it. But for ensembles with high diversity or independent clusterings, this could be used to speed up convergence.
        patience_rounds: int, default=3, number of rounds without improvement that the algorithm will wait until it terminates.
    
    """

    # Add two more rounds for training of classifiers on the final representation.
    max_rounds += 2
    if pretrain_classifiers and not use_clf:
        raise ValueError("use_clf cannot be set to False if pretrain_classifiers is set to True.")

    if subsample_size < batch_size:
        print(f"WARNING: Subsample_size of {subsample_size} is smaller than batch_size of {batch_size}. Will set batch_size=subsample_size.")
        batch_size_to_use = subsample_size
    else:
        batch_size_to_use = batch_size
    evalloader_train = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(*(data, )),
                                            batch_size=batch_size_to_use,
                                            shuffle=False,
                                            drop_last=False)
    
    
    # Classifier Params
    emb_size = model.layers[-1]
    if clf_layers is None:
        # linear classifier
        clf_layers = [emb_size]
    if clf_layers[0] is None:
        clf_layers[0] = emb_size

    # Init Losses (Mean loss uses cluster_dict and is initialized in the loop)
    classifier_loss_fn = torch.nn.CrossEntropyLoss

    if use_rec:
        rec_loss = ReconstructionLoss(loss_fn=torch.nn.MSELoss, weight=regularizer_weight)
    else:
        rec_loss = None

    # Init other params
    # stop if agreement did not improve for specified consecutive patience rounds
    early_stopping = EarlyStopping(patience=patience_rounds, min_delta=stability_threshold*2, verbose=True)
    if n_clusters is None:
        n_clusters = len(set(gt_labels.tolist()))

    avg_agreement_old = np.inf
    avg_agreement_new = np.inf
    best_agreement = 0
    best_round = 0
    agreement_diff = np.inf
    current_val_loss = np.inf
    last_best_path = None
    u_subsample_data = None
    u_subsample_embedding = None
    final_round = False
    sampling_weights = None
    for round_i in range(max_rounds):
        print(f"Start Round {round_i} with Current Agreement: {avg_agreement_new:.5f} with difference: {agreement_diff:.5f} and threshold: {stability_threshold}\n")
        print(f"Current clf_lr: {clf_lr:.5f}, repr_lr: {repr_lr:.5f}")

        print("\nUpdate Clusterings")
        model.eval()
        subsample_data, subsample_gt_labels, subsample_idx = shuffle_dataset(x=data,
                                                                             y=gt_labels,
                                                                             subsample_size=subsample_size,
                                                                             sampling_weights=sampling_weights,
                                                                             return_index=True)
        subsample_embedding = encode_batchwise_from_data(data=subsample_data, model=model, device=device)
        clusterer_dict = clusterer_dict_fn(n_clusters)
        cluster_dict = create_cluster_dict(clusterer_dict, subsample_embedding, subsample_gt_labels, verbose=True)

        if use_unlabelled:
            u_subsample_data = np.delete(data, subsample_idx, axis=0)
            u_subsample_embedding = encode_batchwise_from_data(data=u_subsample_data, model=model, device=device)


        # remove small clusters (size <= 5) to avoid overhead
        cluster_dict, subsample_data, subsample_gt_labels = remove_small_clusters(cluster_dict=cluster_dict, data=subsample_data, labels=subsample_gt_labels, threshold=5)
        # Create train and eval split for early stopping and learning rate scheduler
        cluster_dict_train, cluster_dict_eval, subsample_data_train, subsample_data_eval, subsample_gt_labels_train, subsample_gt_labels_eval = cluster_dict_train_test_split(data=subsample_data, cluster_dict=cluster_dict, n_train=train_split_ratio, gt_labels=subsample_gt_labels)
        subsample_embedding_train = encode_batchwise_from_data(data=subsample_data_train, model=model, device=device)
        subsample_embedding_eval = encode_batchwise_from_data(data=subsample_data_eval, model=model, device=device)

        # Init MeanLoss
        if use_mean:
            # slowly rampup mean loss over the course of training
            # this enforces the agreement over time
            if mean_weight_rampup is not None and mean_weight_rampup > 0:
                mean_weight = get_current_consistency_weight(global_step=round_i,
                                        number_of_batches_in_dataloader=1,
                                        unsup_mixup_consistency_weight=mean_weight_rampup,
                                        consistency_rampup_ends=max_rounds,
                                        consistency_rampup_starts=1,
                                        epoch=round_i)
                print(f"Round {round_i} Mean Weight: {mean_weight}")
            else:
                mean_weight = 1.0
            mean_loss = MeanLoss(soft_assignments=False,
                                 # Only needed if soft_assignments is False
                                 assignment_matrix_dict=get_assignment_matrix_dict(cluster_dict_train, device=device),
                                 assignment_matrix_eval_dict=get_assignment_matrix_dict(cluster_dict_eval, device=device),
                                 weight=mean_weight,
                                 use_clf=use_clf)
            if mean_loss.assignment_matrix_dict is not None:
                mean_loss.init_label_means_dict(embedded=torch.from_numpy(subsample_embedding_train).float().to(device))
        else:
            mean_loss = None
        # init classifiers
        classifier_ensemble = ensemble_class(layers=clf_layers, cluster_label_dict=cluster_dict_train,  eval_cluster_label_dict=cluster_dict_eval, full_cluster_label_dict=cluster_dict)

        if pretrain_classifiers:
            print("\nStart training classifiers")
            train_classifiers(classifier_ensemble=classifier_ensemble,
                            training_iterations=clf_iterations,
                            lr=clf_lr,
                            batch_size=batch_size_to_use,
                            data=subsample_embedding_train,
                            data_eval=subsample_embedding_eval,
                            unlabelled_data=u_subsample_embedding,
                            optimizer_fn=clf_optimizer_fn,
                            classifier_loss_fn=classifier_loss_fn,
                            patience=patience,
                            device=device,
                            print_step=print_step,
                            model_path=model_path+"clf_temp")
            # remove temp file
            os.remove(model_path+"clf_temp")
        
        classifier_ensemble.to(device)
        if pretrain_classifiers:
            print("\nClassifier Train Performance")
            # Evaluate on full data
            prediction_dict = get_prediction_dict(classifier_dict=classifier_ensemble,
                                                model=model,
                                                dataloader=evalloader_train,
                                                device=device,
                                                labels=gt_labels)
        else:
            # If classifiers are not pretrained we cannot use them to predict labels for unclustered data points
            prediction_dict = cluster_dict
        with open(model_path+f"-predictions-round-{round_i}", "wb") as f:
            pickle.dump(prediction_dict, f)
        
        if subsample_embedding.shape[1] == 2:
            if save_plots is not None:
                print("\nclassifiers only: ")
                plot_classifier_probabilities(classifier_ensemble.classifiers, torch.from_numpy(subsample_embedding_eval), gt_labels=subsample_gt_labels_eval, device=device, plot_centers=False,
                                            save=os.path.join(save_plots, f"preupdated_classifier_probabilities_round_{round_i}.png"))
            
        # Recalculate Agreement
        weight_dict = calculate_weights(cluster_dict)

        avg_agreement_new = calculate_avg_agreement(weight_dict)
        agreement_diff = np.abs(avg_agreement_new - avg_agreement_old)


        # save current best ensemble
        if avg_agreement_new > best_agreement or agreement_diff < stability_threshold:
            best_round = round_i
            best_agreement = avg_agreement_new
            best_weight_dict = weight_dict
            best_val_loss = current_val_loss
            if model_path is not None:
                if classifier_ensemble.full_cluster_label_dict is not None:
                    cluster_label_dict = classifier_ensemble.full_cluster_label_dict
                else:
                    cluster_label_dict = classifier_ensemble.cluster_label_dict
                if last_best_path is not None:
                    os.remove(last_best_path)
                last_best_path = model_path+f"_max_agreement_round_{best_round}"
                torch.save({"model":model.state_dict(), "classifiers":classifier_ensemble.state_dict(), "cluster_dict":cluster_label_dict,  "best_round": best_round, "best_agreement": best_agreement, "best_weight_dict": best_weight_dict, "best_repr_val_loss": best_val_loss}, last_best_path)

        # Stop training if stable agreement is reached
        if (agreement_diff < stability_threshold):
            print(f" Stop training at round {round_i} stable agreement reached at {agreement_diff:.5f}, best round was round {best_round} with {best_agreement:.5f}")
            final_round = True
        else:
            avg_agreement_old = avg_agreement_new

        # Stop training if there is no agreement improvement within the specified rounds
        # use negative, because we want to maximize agreement
        early_stopping(-avg_agreement_new)
        if early_stopping.early_stop:
            print(f"Stop training at round {round_i}, due to no improvement over {patience_rounds}. best round was round {best_round} with agreement of {best_agreement}")
            final_round = True
        # If max rounds is reached set final_round to True
        # if pretrain_classifiers is False we already start with the final_round one round early due to
        # int(True)== 1, so that the classifiers are trained in the last round
        if round_i == (max_rounds - (2 + int(not pretrain_classifiers) ) ):
            final_round = True
        
        # Speeds up convergence once the ensemble is already strongly agreeing
        if avg_agreement_new > hardening_agreement_threshold:
            weight_dict = calculate_max_normalized_weights(cluster_dict)
        print("Agreement per prediction: ", weight_dict)

 
        # If loss explodes or becomes none --> reduce lr and retrain model and classifier
        if not final_round:
            print("\nStart updating representation")
            success = False
            max_restarts = 10
            restart_i = 0
            # Backup last model
            torch.save({"model":model.state_dict(), "classifiers":classifier_ensemble.state_dict(), "cluster_dict":cluster_label_dict}, model_path+"_last")
            while(not success):
                try:
                    model, classifier_ensemble, current_val_loss = train_representation(classifier_ensemble=classifier_ensemble,
                                                                                                model=model,
                                                                                                training_iterations=max_iterations,
                                                                                                lr=repr_lr,
                                                                                                batch_size=batch_size_to_use,
                                                                                                data=subsample_data_train.numpy(),
                                                                                                data_eval=subsample_data_eval.numpy(),
                                                                                                unlabelled_data=None if u_subsample_data is None else u_subsample_data.numpy(),
                                                                                                optimizer_fn=repr_optimizer_fn,
                                                                                                use_clf=use_clf,
                                                                                                fix_classifiers=fix_classifiers,
                                                                                                classifier_loss_fn=classifier_loss_fn,
                                                                                                mean_loss=mean_loss,
                                                                                                regularizer_list=[rec_loss],
                                                                                                weight_dict=weight_dict if use_agreement_weighting else None,
                                                                                                patience=patience,
                                                                                                device=device,
                                                                                                print_step=print_step,
                                                                                                model_path=model_path+"repr_temp")
                    # remove temp file
                    os.remove(model_path+"repr_temp")
                    os.remove(model_path+"_last")
                    success = True
                    repr_lr *= repr_lr_decrease_rate
                # if loss exploded or is None then reload old model and reduce the learning rate
                except LossNanError:
                    restart_i += 1
                    repr_lr *= 0.5
                    print("#"*20)
                    print(f"Representation Update Restart: {restart_i} with repr_lr: {repr_lr}")
                    print("#"*20)
                    if restart_i >= max_restarts:
                        raise RuntimeError(f"Representation update has exceeded max_restarts of {max_restarts} with learning rate {repr_lr}. Consider other changes to make training more stable.")
                    # reload last best model
                    sd = torch.load(model_path+"_last")
                    model.load_state_dict(sd["model"])
                    model.to(device)
                    # reinit classifiers
                    classifier_ensemble = ensemble_class(layers=classifier_ensemble.layers, cluster_label_dict=classifier_ensemble.cluster_label_dict,
                                                         eval_cluster_label_dict=classifier_ensemble.eval_cluster_label_dict, full_cluster_label_dict=classifier_ensemble.full_cluster_label_dict)
                    # load last best classifier_ensemble
                    classifier_ensemble.load_state_dict(sd["classifiers"])
                    classifier_ensemble.to(device)
                    if use_mean:
                        # Reinit mean loss
                        mean_loss = MeanLoss(soft_assignments=mean_loss.soft_assignments, use_clf=use_clf,
                                            assignment_matrix_dict=mean_loss.assignment_matrix_dict, assignment_matrix_eval_dict=mean_loss.assignment_matrix_eval_dict)

            model.to(device)
            classifier_ensemble.to(device)
        
        # Sample uncertain data points more often
        if avg_agreement_new > hardening_agreement_threshold:
            sampling_weights = get_classifier_uncertainty_weights(classifier_dict=classifier_ensemble,
                                                model=model,
                                                dataloader=evalloader_train,
                                                device=device,
                                                past_weights=sampling_weights)
        else:
            sampling_weights = None

        if save_plots is not None:
            subsample_embedding = encode_batchwise_from_data(data=subsample_data_train, model=model, device=device)
            if len(cluster_dict_train) > 3:
                for name in ["km", "spectral", "agglo", "gauss", "sc", "agg", "gmm"]:
                    helper_dict = {}
                    for key, value in cluster_dict_train.items():
                        if name in key.lower():
                            helper_dict[key] = value
                    if len(helper_dict) > 0:
                        plot_clusterings(subsample_embedding, helper_dict, subsample_gt_labels_train, save=os.path.join(save_plots, f"{name}_clusterings_round_{round_i}.png"), center_dict=None if mean_loss is None else mean_loss.label_means_dict)
            else:
                plot_clusterings(subsample_embedding, cluster_dict_train, subsample_gt_labels_train, save=os.path.join(save_plots, f"clusterings_round_{round_i}.png"), center_dict=None if mean_loss is None else mean_loss.label_means_dict)
            if subsample_embedding.shape[1] == 2:
                eval_subsample_embedding = encode_batchwise_from_data(data=subsample_data_eval, model=model, device=device)
                plot_classifier_probabilities(classifier_ensemble.classifiers, torch.from_numpy(eval_subsample_embedding), gt_labels=subsample_gt_labels_eval, device=device, plot_centers=False, save=os.path.join(save_plots, f"classifier_probabilities_round_{round_i}.png"))
        if final_round:
            # If classifiers are not pretrained we want to run one more time to get a trained classifier
            if pretrain_classifiers:
                break
            else:
                # In the last round not trained classifiers are finetuned
                # and the clustering is performed in the full space
                pretrain_classifiers = True
    # return last model
    return model, classifier_ensemble
