import torch
from lightning.pytorch import seed_everything
import lightning as L
from utilities.utils import *
from utilities.utils_compression_set import *
from utilities.utils_datasets import *
from utilities.utils_early_stopping import *
from utilities.utils_models import *
from dataset.dataset_loader import load_dataset
import os
import json
import argparse
import wandb
import yaml
from copy import deepcopy
from bounds.pac_bayes_bounds import monte_carlo_sampling, compute_pac_bayes_bound, compute_pac_bayes_disagreement_bound
from multiprocessing import Pool
from pytorch_lightning.loggers import WandbLogger

import sys
sys.path.insert(0, './PBB')
from PBB.pbb.bounds import PBBobj


def pbb_experiments(config, name):
    wandb.init(project=name, config=config)
    logger = WandbLogger(project=name, experiment=wandb.run, prefix="(train)")
    file_name = get_exp_file_name(dict(wandb.config), path="./pbb_logs/")
    model_name = get_model_name_from_config(wandb.config, use_pbb=True)

    seed_everything(wandb.config['seed'], workers=True)
    accelerator = get_accelerator(wandb.config['model_type'])

    # create models, load dataset and split it if necessary
    
    train_set, test_set, collate_fn = load_dataset(wandb.config)
    if wandb.config['prior_size'] != 0.0:
        train_set, validation_set = split_train_validation_dataset(train_set, wandb.config['validation_size'])
        train_set, disagreement_set = split_train_validation_dataset(train_set, wandb.config.get('disagreement_size',0.2))
        train_set, prior_set = split_train_validation_dataset(train_set, wandb.config['prior_size'])
    else:
        train_set, validation_set = split_train_validation_dataset(train_set, wandb.config['validation_size'])
        train_set, disagreement_set = split_train_validation_dataset(train_set, wandb.config.get('disagreement_size',0.2))
        prior_set = None

    # transform = get_data_augmentation_transform(wandb.config['dataset'], "baseline")
    transform = get_data_augmentation_transform(wandb.config['dataset'], "p2l")
    train_set.transform = transform
    if prior_set is not None:
        # prior_set.transform = transform
        prior_set.transform = get_data_augmentation_transform(wandb.config['dataset'], "p2l")

    trainset_loader = get_dataloader(dataset=train_set, batch_size=wandb.config['batch_size'], collate_fn=collate_fn)
    valset_loader = get_dataloader(dataset=validation_set, batch_size=wandb.config['batch_size'], collate_fn=collate_fn)
    disagreement_loader = get_dataloader(dataset=disagreement_set, batch_size=wandb.config['batch_size'], collate_fn=collate_fn)
    test_loader = get_dataloader(dataset=test_set, batch_size=wandb.config['batch_size'], collate_fn=collate_fn)

    if wandb.config['prior_size'] == 0.0:
        model = create_model(wandb.config)
    else:
        file_path = "./prior_models/"
        if not os.path.isdir(file_path):
            os.mkdir(file_path)
        
        prior_model_name = f"{wandb.config['dataset']}_{wandb.config['model_type']}_{wandb.config['prior_size']}_{wandb.config['pretraining_lr']}_{wandb.config['pretraining_epochs']}_{wandb.config['seed']}.ckpt"
        file_path = file_path + prior_model_name
        if os.path.isfile(file_path):
            model = load_pretrained_model(file_path, wandb.config)
        else:
            model = create_model(wandb.config)
            prior_loader= get_dataloader(dataset=prior_set, batch_size=wandb.config['batch_size'] , collate_fn=collate_fn)
            prior_trainer = get_trainer(accelerator=accelerator, max_epochs=wandb.config['pretraining_epochs'], logger=logger)
            prior_trainer.fit(model=model, train_dataloaders=prior_loader, val_dataloaders=trainset_loader)
            prior_trainer.save_checkpoint(file_path)
    
    # create probabilistic model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    min_max_vals = get_min_max_loss(wandb.config['min_probability'], wandb.config['n_classes'], wandb.config['clamp_method'])

    bound = PBBobj("fclassic", wandb.config['min_probability'], model.model.n_classes, wandb.config['delta'],
                    wandb.config['delta'], wandb.config["mc_samples"], wandb.config["kl_penalty"],
                    device, n_posterior = len(train_set), n_bound=len(train_set))
    
    model = create_probabilistic_model(wandb.config, model, bound, min_max_vals)
    model.set_sampling_mode(True)

    update_learning_rate(model, wandb.config.get('training_lr', None), len(trainset_loader))
    update_clamping_method(model, wandb.config['clamp_method'])
    if wandb.config['clamping']:
        add_clamping_to_model(model, config=wandb.config)
    
    trainer = get_trainer(accelerator=accelerator,
                        max_epochs=wandb.config['max_epochs'], 
                        callbacks=[EarlyStopping(monitor="validation_error", patience=10)],
                            logger=logger)

    trainer.fit(model=model, train_dataloaders=trainset_loader, val_dataloaders=valset_loader)

    train_set.transform = validation_set.transform
    trainset_loader = get_dataloader(dataset=train_set, batch_size=wandb.config['batch_size'], collate_fn=collate_fn)

    train_results = trainer.validate(model=model, dataloaders=trainset_loader)
    validation_results = trainer.validate(model=model, dataloaders=valset_loader)
    test_results = trainer.test(model, dataloaders=test_loader)
    
    logger._prefix = ""
    information_dict = {}

    information_dict['train_set_size'] = len(train_set)
    information_dict['validation_set_size'] = len(validation_set)
    information_dict['test_set_size'] = len(test_set)

    information_dict['complement_error'] = train_results[0]['validation_error']
    information_dict['validation_error'] = validation_results[0]['validation_error']
    information_dict['test_error'] = test_results[0]['test_error']

    information_dict['complement_loss'] = train_results[0]['validation_loss']
    information_dict['validation_loss'] = validation_results[0]['validation_loss']
    information_dict['test_loss'] = test_results[0]['test_loss']

    error_mc, loss_mc = monte_carlo_sampling(model.model, wandb.config['mc_samples'], trainset_loader, device, wandb.config)
    information_dict['error_mc'] = error_mc.item()
    information_dict['loss_mc'] = loss_mc.item()

    bound_error, kl_val = compute_pac_bayes_bound(model, error_mc.item(), 0.0 , 1.0, wandb.config['delta'], len(train_set), wandb.config['mc_samples'])
    bound_loss, _ = compute_pac_bayes_bound(model, loss_mc.item(), min_max_vals[0], min_max_vals[1], wandb.config['delta'], len(train_set), wandb.config['mc_samples'])

    information_dict['kl_val'] = kl_val
    information_dict['kl_bound'] = bound_error
    information_dict['CE_kl_bound'] = bound_loss

    sum_disagreement, sum_loss_disag, sum_softmax_disag = compute_pac_bayes_disagreement(model, wandb.config['mc_samples'],
                                                                                        disagreement_loader, device,
                                                                                        wandb.config, information_dict)
    compute_pac_bayes_disagreement_bound(sum_disagreement, sum_loss_disag, sum_softmax_disag,
                                        min_max_vals, len(disagreement_set), wandb.config['delta'],
                                         wandb.config['min_probability'], wandb.config['n_classes'], information_dict)

    file_path = "./pbb_trained_models/"
    if not os.path.isdir(file_path):
        os.mkdir(file_path)
    
    file_path = file_path + model_name
    trainer.save_checkpoint(file_path)


    wandb.log(information_dict)
    information_dict['config'] = dict(wandb.config)

    # save the experiment informations in a json
    if not os.path.isdir("./pbb_logs"):
        os.mkdir("./pbb_logs")

    with open(file_name, "w") as outfile: 
        json.dump(information_dict, outfile)

    wandb.finish()

def hyperparameter_loop(list_of_sweep_configs, dataset_config):
    for sweep_config_ in list_of_sweep_configs:
        exp_config = sweep_config_ | dataset_config
        config_name = get_exp_file_name(exp_config, path="./pbb_logs/")
        if not os.path.isfile(config_name):
            exp_name = 'pbb_' + dataset_config['dataset'] + "_ckpt"
            if dataset_config.get('n_classes', -1) == 2 and dataset_config['dataset'] == "mnist":
                exp_name += str(dataset_config['first_class']) + str(dataset_config['second_class'])
            pbb_experiments(exp_config, name=exp_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default="cifar10", help="Name of the dataset")
    args = parser.parse_args()

    sweep_config_name = "./configs/experiment_configs/pbb/" + args.dataset + ".yaml"
    with open(sweep_config_name) as file:
        sweep_configuration = yaml.safe_load(file)

    params_config_name = "./configs/dataset_configs/" + args.dataset + ".yaml"
    with open(params_config_name) as file:
        config = yaml.safe_load(file)

    # correct types of entries to make sure all floats/ints are parsed as such
    for key, value in config.items():
        config[key] = correct_type_of_entry(value)


    list_of_configs = create_all_configs(sweep_configuration)
    if config.get('n_classes', -1) != 2 or config['dataset'] != "mnist":
        hyperparameter_loop(list_of_configs, config)
    else:
        if not isinstance(config['first_class'], list):
            config['first_class'] = [config['first_class']]
            config['second_class'] = [config['second_class']]
        
        for idx in range(len(config['first_class'])):
            new_config = deepcopy(config)
            new_config['first_class'] = config['first_class'][idx]
            new_config['second_class'] = config['second_class'][idx]
            hyperparameter_loop(list_of_configs, new_config)
