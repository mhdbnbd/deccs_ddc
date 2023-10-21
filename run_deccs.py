import os
import argparse
import pickle
import torch.utils.data
import torch
import numpy as np
import pandas as pd
import torchvision
from collections import OrderedDict
from utils import cmd_interface
from utils.utils import setup_directory, detect_device
from utils.data import DatasetsImages
from utils.data import DatasetsUci
from utils.data import DatasetsSynthetic
from utils.data import fetch_data_set_by_name, shuffle_dataset, print_data_statistics
from model.autoencoder import Autoencoder
from model.stacked_ae import StackedAE
from model.ensemble import ClassifierEnsemble
from utils.train import alternate_optimization
from copy import deepcopy
from utils.utils import random_seed, flatten_tensor
from utils.ensemble import get_prediction_dict, eval_prediction_dict
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from utils.clustering import ClustererEnsembles
from utils.utils import write_json, cluster_accuracy

# remote test 

def main(flags):
    if flags.json is not None:
        setting = flags.json
    else:
        setting = vars(flags)

    if setting["data"] == DatasetsUci.__name__:
        dataset_list = DatasetsUci.ALL
    elif setting["data"] == DatasetsImages.__name__:
        dataset_list = DatasetsImages.ALL
    elif setting["data"] ==  DatasetsSynthetic.__name__:
        dataset_list = DatasetsSynthetic.ALL
    else:
        dataset_list = [setting["data"]]
    
    if setting["ensemble_type"] in ClustererEnsembles.ALL:
        clusterer_dict_fn = ClustererEnsembles.DICT[setting["ensemble_type"]]
    else:
        raise ValueError(f"Specified ensemble_type is not implemented, submitted value is {setting['ensemble_type']}")
    ensemble_class = ClassifierEnsemble
    
    # init optimizer
    optimizer_fn = lambda params, lr: torch.optim.SGD(params, lr, momentum=setting["momentum"], weight_decay=setting["weight_decay"])
    device = torch.device(setting["device"])
    print("Using device: ", device)
    for data_set_name_i in dataset_list:
        result_dir = os.path.join(setting["results_dir"], data_set_name_i)
        models_dir = os.path.join(result_dir, "pretrained_aes")
        # setup deep ensemble directory
        ensemble_str = "deccs"
        if setting["ensemble_type"] is not None:
            ensemble_str += "-"+setting["ensemble_type"]
        if setting["use_rec"]:
            ensemble_str += "-rec"
        if setting["use_mean"]:
            ensemble_str += "-mean"
        if not setting["use_clf"]:
            ensemble_str += "-no_clf"
        if setting["use_unlabelled"]:
            ensemble_str += "-use_unlabelled"
        if setting["mean_weight_rampup"] is not None:
            ensemble_str += f"-mean_weight_rampup_{setting['mean_weight_rampup']:.1f}"
        if not setting["pretrain_classifiers"]:
            ensemble_str += "-no_clf_pretraining"
        if not setting["use_agreement_weighting"]:
            ensemble_str += "-no_agreement_weighting"
        if setting["model_type"] is not None: 
            ensemble_str += f"-{setting['model_type']}"
        if setting["output_dir"] is None:
            ensemble_dir = os.path.join(result_dir, data_set_name_i, ensemble_str)
        else:
            ensemble_dir = os.path.join(setting["output_dir"], data_set_name_i, ensemble_str)

        setup_directory(ensemble_dir)
        if setting["generate_plots"]:
            plot_dir = os.path.join(ensemble_dir, "plots")
            setup_directory(plot_dir)
        else:
            plot_dir = None
        if setting["hp_dir"] is not None:
            # load best hyperparameter setting for the autoencoder
            hp_dict = torch.load(os.path.join(setting["hp_dir"], data_set_name_i, "setting.dict"))["best_params"]
            # override parameters in setting dict
            for key, value in hp_dict.items():
                if key not in ["max_iterations", "data", "hp_dir"]:
                    # set representation learning rate to the autoencoder learning rate if it is None
                    if (setting["repr_lr"] is None) and key == "lr":
                        setting["repr_lr"] = value 
                    setting[key] = value


        
        if setting["agreement_threshold"] is None:
            # Set to -np.inf so that we will train until max_round is reached
            setting["agreement_threshold"] = -np.inf
        
        print("######################")
        print("Load data set: ", data_set_name_i)
        print("######################")
        data_train, labels_train = fetch_data_set_by_name(data_set_name_i, train=None, verbose=False)
        data_test, labels_test = fetch_data_set_by_name(data_set_name_i, train=None, verbose=False)



        print(f"{data_set_name_i}-Train:")
        print_data_statistics(data_train, labels_train)
        print(f"{data_set_name_i}-Test:")
        print_data_statistics(data_test, labels_test)

        n_clusters = len(set(labels_train.tolist()))
        print("number of clusters: ", n_clusters)

        trainloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(*(data_train, labels_train)),
                                        batch_size=setting["batch_size"],
                                        shuffle=False,
                                        drop_last=False)

        testloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(*(data_test, labels_test)),
                                                batch_size=setting["batch_size"],
                                                shuffle=False,
                                                drop_last=False)

        # Set ratio based on the data set size if it is None
        if setting["subsample_size_ratio"] is None:
            if data_train.shape[0] <= 11000:
                subsample_size_ratio = 0.5
            else:
                subsample_size_ratio = 0.08
        
       
        # Make a copy of ae_layers so we can override it in case the last layer is None
        ae_layers = deepcopy(setting["ae_layers"])
        if ae_layers[-1] is None:
            ae_layers[-1] = n_clusters
        # Add first input to hidden layer
        ae_layers = [data_train.shape[1]] + ae_layers

        # Setup metrics
        metrics_fn_dict = OrderedDict({"nmi":normalized_mutual_info_score, "acc":cluster_accuracy, "ari":adjusted_rand_score})
        # Setup scores
        scores = OrderedDict({"nmi_train": [], "acc_train": [], "ari_train": [],
                              "nmi_test": [], "acc_test": [], "ari_test": []})
        all_scores = OrderedDict({name:deepcopy(scores) for name,_ in clusterer_dict_fn(n_clusters=n_clusters).items()})
        
        # save used setting in ensemble folder
        write_json(fname=os.path.join(ensemble_dir, "settings.json"), dictionary=setting)

        labels_train = labels_train.numpy()
        labels_test = labels_test.numpy()
        for model_index in range(0, setting["nr_models"]):
            print(f"\nDataSet {data_set_name_i}: Start clustering ae {model_index}/{setting['nr_models']}")
            
            ensemble_model_name = f"ensemble-model-idx-{model_index}.pth"
            model_path = os.path.join(ensemble_dir,ensemble_model_name)
            
            if setting["model_type"] is None:
                model = Autoencoder(layers=ae_layers, dropout=setting["dropout"])
                model_name = f"ae-model-idx-{model_index}.pth"
                sd = torch.load(os.path.join(models_dir, model_name), map_location=device)
                print(f"Load model {os.path.join(models_dir, model_name)}")
                model.load_state_dict(sd["model"])
            elif "stacked_ae" == setting["model_type"]:
                model = StackedAE(layers=ae_layers, dropout=setting["dropout"])
                model_name = f"ae-model-idx-{model_index}.pth"
                sd = torch.load(os.path.join(models_dir, model_name), map_location=device)
                print(f"Load model {os.path.join(models_dir, model_name)}")
                model.load_state_dict(sd["model"])
            model = model.to(device)
            # Returns last model and ensemble
            model, classifier_ensemble = alternate_optimization(model=model,
                                                    data=data_train,
                                                    gt_labels=labels_train,
                                                    clusterer_dict_fn=clusterer_dict_fn,
                                                    ensemble_class=ensemble_class,
                                                    max_rounds=setting["max_rounds"],
                                                    max_iterations=setting["max_iterations"],
                                                    clf_iterations=setting["max_iterations"],
                                                    clf_lr=setting["clf_lr"],
                                                    repr_lr=setting["repr_lr"],
                                                    clf_optimizer_fn=optimizer_fn,
                                                    repr_optimizer_fn=optimizer_fn,
                                                    train_split_ratio=setting["train_split_ratio_ensemble"],
                                                    subsample_size=int(subsample_size_ratio*data_train.shape[0]),
                                                    agreement_threshold=setting["agreement_threshold"],
                                                    use_agreement_weighting=setting["use_agreement_weighting"],
                                                    repr_lr_decrease_rate=setting["repr_lr_decrease_rate"],
                                                    patience=setting["patience"],
                                                    use_clf=setting["use_clf"],
                                                    use_rec=setting["use_rec"],
                                                    use_mean=setting["use_mean"],
                                                    mean_weight_rampup=setting["mean_weight_rampup"],
                                                    clf_layers = setting["clf_layers"],
                                                    fix_classifiers=setting["fix_classifiers"],
                                                    use_unlabelled=setting["use_unlabelled"],
                                                    regularizer_weight=setting["regularizer_weight"],
                                                    batch_size=setting["batch_size"],
                                                    n_clusters=n_clusters,
                                                    pretrain_classifiers=setting["pretrain_classifiers"],
                                                    device=device,
                                                    model_path=model_path,
                                                    save_plots=plot_dir,
                                                    print_step=setting["print_step"])
        

            # Save last stable model
            torch.save({"model":model.state_dict(), "classifiers":classifier_ensemble.state_dict(), "cluster_dict":classifier_ensemble.cluster_label_dict}, model_path)
            
            train_prediction_dict = get_prediction_dict(classifier_dict=classifier_ensemble,
                                                        model=model,
                                                        dataloader=trainloader,
                                                        device=device,
                                                        labels=None)
            all_scores = eval_prediction_dict(prediction_dict=train_prediction_dict, 
                                                    metrics_fn_dict=metrics_fn_dict,
                                                    labels=labels_train,
                                                    metric_suffix="_train",
                                                    res_dict=all_scores)
            
            test_prediction_dict = get_prediction_dict(classifier_dict=classifier_ensemble,
                                                        model=model,
                                                        dataloader=testloader,
                                                        device=device,
                                                        labels=None)
            all_scores = eval_prediction_dict(prediction_dict=test_prediction_dict, 
                                                    metrics_fn_dict=metrics_fn_dict,
                                                    labels=labels_test,
                                                    metric_suffix="_test",
                                                    res_dict=all_scores)
                            
            with open(os.path.join(ensemble_dir, "ensemble-"+model_name.split(".")[0]+".labels_train"), "wb") as f:
                pickle.dump(train_prediction_dict, f)
            with open(os.path.join(ensemble_dir, "ensemble-"+model_name.split(".")[0]+".labels_test"), "wb") as f:
                pickle.dump(test_prediction_dict, f)
            del model
            del classifier_ensemble
        for name, scores in all_scores.items():
            print(f"Scores {name}-Ensemble:")
            for score_name, score_i in scores.items():
                print(f"{score_name}: {np.mean(score_i):.5f} with std: {np.std(score_i)}")
        with open(os.path.join(ensemble_dir, "cluster_scores.pkl"), "wb") as f:
            pickle.dump(all_scores, f)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str,
                    default="results",
                    help="Results Directory to load models from")
    parser.add_argument("--output_dir", type=str,
                default=None,
                help="Output Directory to save results to. If None, will be the same as results_dir.")
    parser.add_argument("--hp_dir", type=str,
            default="hp_results",
            help="Directory to load hp settings from")
    parser.add_argument("--ensemble_type", type=str,
        default=None,
        help=f"Define which cluster ensemble from utils.clustering.ClustererEnsemble you want to use. Has to be one of {ClustererEnsembles.ALL}.")
    parser.add_argument("--generate_plots",
                type=cmd_interface.str2bool,
                default="False",
                help="Set True if plots should be generated.")
    parser.add_argument("--data", type=str,
                        help=f"Datasets to use: should be one of {DatasetsUci.__name__} or {DatasetsImages.__name__}")
    parser.add_argument("--model_type", type=str,
                        default="None",
                        help="model type to use e.g. stackeAE, if None then autoencoder is used")
    parser.add_argument("--ae_layers", type=list,
                        default=None,
                        help="Define number of layers to be used for the encoder (if last layer is set to None it will be set to be equal to the number of clusters), the decoder is the mirrored version.")
    parser.add_argument("--clf_layers", type=list,
                        default=None,
                        help="Define number of layers to be used for the encoder of the classifier (if last layer is set to None it will be set to be equal to the number of clusters). \
                              If None then a linear classifier will be used.")
    parser.add_argument("--nr_models", type=int,
                    default=10,
                    help="Number of models")      
    parser.add_argument("--train_split_ratio_ensemble", type=float,
        default=0.5,
        help="ratio of training data to evaluation data for the classifiers in the ensemble.")
    parser.add_argument("--batch_size", type=int,
            default=256,
            help="Minibatch size")
    parser.add_argument("--subsample_size_ratio", type=float,
        default=None,
        help="Ratio of training data points that are used for performing the clustering algorithms on. If None, then 0.08 is used for data sets > 11,000 and 0.5 else.")
    parser.add_argument("--agreement_threshold", type=float,
            default=1e-5,
            help="Agreement threshold to determine when to stop training. If None, than algorithm will be trained until max_rounds are reached.")
    parser.add_argument("--patience", type=int,
        default=20,
        help="patience parameter for early stopping of classifier and representation training")
    parser.add_argument("--use_agreement_weighting",
                    type=cmd_interface.str2bool,
                    default="True",
                    help="Set True losses should be weighted by agreement")
    parser.add_argument("--pretrain_classifiers",
                        type=cmd_interface.str2bool,
                        default="True",
                        help="Set True if classifiers should be pretrained")
    parser.add_argument("--use_rec",
                        type=cmd_interface.str2bool,
                        default="False",
                        help="Set True if reconstruction error should be used as regularizer")
    parser.add_argument("--use_clf",
                        type=cmd_interface.str2bool,
                        default="True",
                        help="Set True if classifiers should be used to constrain the embedded space")
    parser.add_argument("--mean_weight_rampup",
            type=float,
            default=None,
            help=" if set to a value > 0, it will rampup a weighting parameter over the course of training until half of max_rounds and then the mean_weight_rampup value is reached.")
    parser.add_argument("--use_mean",
                    type=cmd_interface.str2bool,
                    default="True",
                    help="Set True if the cluster mean should be used as cluster compression loss")
    parser.add_argument("--fix_classifiers",
                type=cmd_interface.str2bool,
                default="False",
                help="Set True if the classifiers should not be updated during the representation update")
    parser.add_argument("--use_unlabelled",
            type=cmd_interface.str2bool,
            default="False",
            help="Set True if unlabelled data should be used.")
    parser.add_argument("--regularizer_weight",
            type=float,
            default=1.0,
            help="Weight of regularizer loss term for reconstruction loss it is set to 1.")
    parser.add_argument("--device",
            type=cmd_interface.str2device,
            default="cpu",
            help="Specify device")
    parser.add_argument("--clf_lr", type=float,
            default=1e-2,
            help="classifier learning rate")
    parser.add_argument("--momentum", type=float,
        default=0.9,
        help="momentum parameter for SGD")
    parser.add_argument("--repr_lr", type=float,
        default=1e-3,
        help="representation learning rate")
    parser.add_argument("--repr_lr_decrease_rate", type=float,
        default=1e-3,
        help="representation learning rate decrease after each alternation")
    parser.add_argument("--weight_decay", type=float,
        default=0,
        help="amount of weight decay to be used")
    parser.add_argument("--dropout", type=float,
        default=None,
        help="Amount of droput to be used. If None, dropout will be 0.")
    parser.add_argument("--max_iterations", type=int,
                default=None,
                help="Maximum number of iterations to train.")
    parser.add_argument("--print_step", type=int,
        default=2500,
        help="determines after which iterations the loss should be printed")
    parser.add_argument('--json',
                    type=cmd_interface.json2dict,
                    default=None,
                    help='Path to a json file to specify the flags')
    flags, unparsed = parser.parse_known_args()
   
    main(flags)
