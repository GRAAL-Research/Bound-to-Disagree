import torch
from lightning.pytorch import seed_everything
import lightning as L
from utilities.utils import *
from utilities.utils_compression_set import *
from utilities.utils_datasets import *
from utilities.utils_early_stopping import *
from utilities.utils_models import *
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from dataset.dataset_loader import load_dataset
from bounds.classical_bounds import compute_classical_compression_bounds,\
    brute_force_binomial_tail_inversion, binomial_approximation
from bounds.real_valued_bounds import compute_real_valued_bounds
import os
import json
import argparse
import wandb
import datetime
from functools import partial
import yaml
from copy import deepcopy
from pytorch_lightning.loggers import WandbLogger
import DeepCore.deepcore.methods as methods
from multiprocessing import Pool
import math

def compute_norm_based_bounds(config):
    """
    Code adapted from : https://github.com/Lianga2000/NormBasedBounds
    """
    gamma = config.get('gamma', 1)

    file_path = "./norm_based_bounds/"
    if not os.path.isdir(file_path):
        os.mkdir(file_path)

    file_name = f"{config['dataset']}_{config['model_type']}_{config['seed']}.json"
    file_name = file_path + file_name
    if os.path.isfile(file_name):
        with open(file_name) as json_file:
            d = json.load(json_file)
            print(d['bound'])
            return 

    best_model = get_best_model(config['dataset'], config['model_type'], config['seed'], None)

    seed_everything(config['seed'], workers=True)
    train_set, _, collate_fn = load_dataset(config)

    train_set, _ = split_train_validation_dataset(train_set, config['validation_size'])
    train_set, _ = split_train_validation_dataset(train_set, config.get('disagreement_size',0.2))
    
    trainset_loader = get_dataloader(dataset=train_set, batch_size=config['batch_size'], collate_fn=collate_fn)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    pixel_sums = torch.zeros(train_set[0][0].size())
    margin_error = 0
    error = 0
    best_model.eval()
    best_model.to(device)
    with torch.no_grad():
        for _,  (x, y) in enumerate(tqdm(trainset_loader, desc="Training error")):
            x, y = x.to(device), y.to(device)
            pixel_sums += torch.pow(x, 2).sum(0).cpu()
            y_hat = best_model.model(x)

            error += (torch.argmax(y_hat, dim=1) != y).sum().item()

            logits_class = y_hat[torch.arange(y_hat.size(0)), y]

            mask = torch.ones(y_hat.shape, dtype=torch.bool)
            mask[torch.arange(y_hat.size(0)), y] = False
            y_hat_no_class = y_hat[mask].reshape(y_hat.shape[0],-1)
            logits_other = y_hat_no_class.max(1).values
            margin_error += (logits_other + gamma >= logits_class).sum().item()

    best_model.cpu()
    rho = 1
    depth = 0
    deg_prod = 1
    if config['model_type'] == "cnn":
        for _, val in best_model.model.cnn_layers.named_children():
            if isinstance(val, torch.nn.modules.conv.Conv2d):
                rho *= torch.linalg.norm(val.weight).detach()
                deg_prod *= val.kernel_size[0]**2
                depth += 1
        for _, val in best_model.model.linear_layers[:-1].named_children():
            if isinstance(val, torch.nn.modules.linear.Linear):
                rho *= torch.linalg.norm(val.weight).detach()
                deg_prod *= val.in_features
                depth += 1
        rho *= torch.linalg.norm(best_model.model.linear_layers[-1].weight).detach()
        depth += 1
    elif config['model_type'] == "resnet":
        for _, val in best_model.model.resnet.named_modules():
            if isinstance(val, torch.nn.modules.conv.Conv2d):
                rho *= torch.linalg.norm(val.weight).detach()
                deg_prod *= val.kernel_size[0]**2
                depth += 1
            elif isinstance(val, torch.nn.modules.linear.Linear):
                rho *= torch.linalg.norm(val.weight).detach()
                depth += 1
    else:
        raise NotImplementedError(f"The norm based bound for {config['model_type']} model types.")


    mult1 = 2 * math.sqrt(2) * (rho.item() + 1) / (gamma * len(train_set))
    mult2 = 1 + math.sqrt(2 * (depth * math.log(2) + math.log(deg_prod) + math.log(config['n_classes'])))
    mult3 = math.sqrt(deg_prod * pixel_sums.max())
    sum1 = 3 * math.sqrt(math.log((2/config['delta']) * (rho + 2)**2)/(2 * len(train_set)))

    information_dict = {}
    information_dict['error'] = error/len(train_set)
    information_dict['margin_error'] = margin_error/len(train_set)
    information_dict['bound_over_gap'] = mult1 * mult2 * mult3 + sum1
    information_dict['bound'] = information_dict['margin_error'] + information_dict['bound_over_gap']

    with open(file_name, "w") as outfile: 
        json.dump(information_dict, outfile)
    print(information_dict['bound'])
    
if __name__ == "__main__":
    for dataset in ['mnist', 'cifar10']:
        for model_type in ['cnn', 'resnet']:
            for seed in [1,2,3,4,42]:
                if dataset == "mnist" and model_type == "resnet":
                    continue

                config = {
                    'dataset': dataset,
                    'model_type': model_type,
                    'seed':seed,
                    'n_classes': 10,
                    'prior_size':0.0,
                    'validation_size':0.1,
                    'disagreement_size':0.2,
                    'batch_size':32,
                    'delta':0.01
                }
                compute_norm_based_bounds(config)
 
