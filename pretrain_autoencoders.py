import os
import argparse
import torch.utils.data
import torch
import numpy as np
import pandas as pd
import torchvision
from utils import cmd_interface
from utils.utils import setup_directory, detect_device
from utils.data import DatasetsImages
from utils.data import DatasetsUci
from utils.data import DatasetsSynthetic
from utils.data import fetch_data_set_by_name, shuffle_dataset, print_data_statistics
from model.autoencoder import Autoencoder
from model.stacked_ae import StackedAE
from utils.train import validate_autoencoder
from copy import deepcopy
from utils.utils import random_seed
#test luke connection


def pretrain(flags):
    try:
        setting = flags.json
    except:
        if isinstance(flags, dict):
            setting = flags
        else:
            setting = vars(flags)
    print("settings: ", setting)
    
    # fixed settings
    loss_fn = torch.nn.MSELoss
    optimizer_fn = torch.optim.Adam
    if setting["data"] == DatasetsUci.__name__:
        dataset_list = DatasetsUci.ALL
    elif setting["data"] == DatasetsImages.__name__:
        dataset_list = DatasetsImages.ALL
    elif setting["data"] ==  DatasetsSynthetic.__name__:
        dataset_list = DatasetsSynthetic.ALL
    else:
        dataset_list = [setting["data"]]
    # set device to train on
    device = torch.device(setting["device"])
    print("Using device: ", device)
    rec_losses_train_dict = {}
    rec_losses_eval_dict = {}
    for data_set_name_i in dataset_list:
        # setup data and results directories
        result_dir = os.path.join(setting["results_dir"], data_set_name_i)
        setup_directory(result_dir)
        models_dir = os.path.join(result_dir, "pretrained_aes")
        setup_directory(models_dir)
        if setting["hp_dir"] is not None:
            # load best hyperparameter setting
            hp_path = os.path.join(setting["hp_dir"], data_set_name_i,  "setting.dict")
            hp_dict = torch.load(hp_path)["best_params"]
            # override parameters in setting dict
            for key, value in hp_dict.items():
                if key not in ["max_iterations", "data", "hp_dir"]:
                    setting[key] = value
            print("settings_new: ", setting)

        print("######################")
        print("Load data set: ", data_set_name_i)
        print("######################")
        # train=None means we return the joined train and test set
        data, labels = fetch_data_set_by_name(data_set_name_i, train=None, verbose=False)
        data_train, train_index  = shuffle_dataset(x=data,
                                    subsample_size=int(setting["train_split"]*data.shape[0]),
                                    return_index=True,
                                    # Always use the same split
                                    random_state=0)
        data_eval = torch.from_numpy(np.delete(data.numpy(), train_index, axis=0))

        print(f"{data_set_name_i}-Train:")
        print_data_statistics(data_train)
        print(f"{data_set_name_i}-Eval:")
        print_data_statistics(data_eval)
        n_clusters = len(set(labels.tolist()))
        print("number of clusters: ", n_clusters)
        
        # Make a copy of ae_layers so we can override it in case the last layer is None
        layers = deepcopy(setting["ae_layers"])
        if layers[-1] is None:
            layers[-1] = n_clusters
        # Add first input to hidden layer
        layers = [data_train.shape[1]] + layers

        np_seeds = np.random.randint(900000, size=setting["nr_aes"])
        rec_losses_train = []
        rec_losses_eval = []
        for ae_index in range(0, setting["nr_aes"]):
            print(f"\nDataSet {data_set_name_i}: Start training ae {ae_index}/{setting['nr_aes']-1}")
            # can be used for debugging
            # random_seed(np_seeds[ae_index])
            if setting["model_type"] is None:
                model = Autoencoder(layers=layers, dropout=setting["dropout"])
            elif setting["model_type"] == "stacked_ae":
                model = StackedAE(layers=layers, dropout=setting["dropout"])

            model_name = f"ae-model-idx-{ae_index}.pth"

            trainloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(*(data_train, )),
                                                    batch_size=setting["batch_size"],
                                                    shuffle=True,
                                                    drop_last=False)

            trainloader_eval = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(*(data_train, )),
                                                    batch_size=setting["batch_size"],
                                                    shuffle=False,
                                                    drop_last=False)

            evalloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(*(data_eval, )),
                                                    batch_size=setting["batch_size"],
                                                    shuffle=False,
                                                    drop_last=False)

            model.to(device)
            model.fit(loss_fn=loss_fn(),
                    lr=setting["lr"],
                    optimizer_fn=lambda params, lr: optimizer_fn(params, lr, weight_decay=setting["weight_decay"]),
                    training_iterations=setting["max_iterations"], 
                    dataloader=trainloader,
                    evalloader=evalloader,
                    model_path=os.path.join(models_dir, model_name), 
                    patience=setting["patience"],
                    device=device)
            
            # Load Best Model
            sd = torch.load(os.path.join(models_dir, model_name))
            model.load_state_dict(sd["model"])
            
            train_loss = validate_autoencoder(model, trainloader_eval, device=device, loss_fn=loss_fn()).item()
            eval_loss = validate_autoencoder(model, evalloader, device=device, loss_fn=loss_fn()).item()

            rec_losses_train.append(train_loss)
            rec_losses_eval.append(eval_loss)
            print(f"FINAL Reconstruction Loss: TRAIN {train_loss:.5f}, EVAL {eval_loss:.5f}")
            del model
        rec_losses_train_dict[data_set_name_i] = rec_losses_train
        rec_losses_eval_dict[data_set_name_i] = rec_losses_eval

        print(f"AVG Reconstruction Loss: TRAIN {np.mean(rec_losses_train):.5f} with std: {np.std(rec_losses_train)}")
        print(f"AVG Reconstruction Loss: EVAL {np.mean(rec_losses_eval):.5f} with std: {np.std(rec_losses_eval)}")
        

        pd.DataFrame({"rec_losses_train": rec_losses_train,
                      "rec_losses_eval": rec_losses_eval}).to_csv(os.path.join(models_dir, f"scores.csv"), sep=";", index=False)

    return rec_losses_train_dict, rec_losses_eval_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str,
                    default="results",
                    help="Results Directory")
    parser.add_argument("--hp_dir", type=str,
                default="hp_results",
                help="Directory to load results of hyperparameter selection from")
    parser.add_argument("--data", type=str,
                        help=f"Datasets to use: should be one of {DatasetsUci.__name__} or {DatasetsImages.__name__}")
    parser.add_argument("--ae_layers", type=list,
                        default=None,
                        help="Define number of layers to be used for the encoder (if last layer is set to None it will be set to be equal to the number of clusters), the decoder is the mirrored version.")
    parser.add_argument("--nr_aes", type=int,
                    default=10,
                    help="Number of autoencoder to pretrain")      
    parser.add_argument("--train_split", type=float,
        default=0.9,
        help="ratio of training data to evaluation data.")
    parser.add_argument("--device",
            type=cmd_interface.str2device,
            default="cpu",
            help="Specify device")
    parser.add_argument("--batch_size", type=int,
            default=256,
            help="Minibatch size")
    parser.add_argument("--patience", type=int,
        default=10,
        help="patience parameter for early stopping")
    parser.add_argument("--dropout", type=float,
        default=None,
        help="Amount of droput to be used. If None, dropout will be 0.")
    parser.add_argument("--lr", type=float,
            default=1e-3,
            help="learning rate")
    parser.add_argument("--weight_decay", type=float,
        default=1e-8,
        help="amount of weight decay to be used")
    parser.add_argument("--max_iterations", type=int,
                default=None,
                help="Maximum number of iterations to train")
    parser.add_argument('--json',
                    type=cmd_interface.json2dict,
                    default=None,
                    help='Path to a json file to specify the flags')
    flags, unparsed = parser.parse_known_args()
   
    pretrain(flags)
