import torch
from lightning.pytorch import seed_everything
import lightning as L
from utilities.utils import *
from utilities.utils_compression_set import *
from utilities.utils_datasets import *
from utilities.utils_early_stopping import *
from utilities.utils_models import *
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from bounds.p2l_bounds import compute_all_p2l_bounds
from bounds.real_valued_bounds import compute_real_valued_bounds
from bounds.classical_bounds import compute_classical_compression_bounds
from dataset.dataset_loader import load_dataset
import os
import json
import argparse
import wandb
import numpy as np
from copy import deepcopy
import yaml

CUDA_DEVICE = 0


def p2l_experiments(config, name='p2l'):
    wandb.init(project=name, config=config)
    seed_everything(wandb.config['seed'], workers=True)

    # constants to be used later 
    STOP = torch.log(torch.tensor(2))
    batch_size = wandb.config['batch_size'] 
    n_sigma = 1
    information_dict = {}
    accelerator = get_accelerator(wandb.config['model_type'])
    device = torch.device(f'cuda:{CUDA_DEVICE}' if torch.cuda.is_available() else 'cpu')

    file_dir = get_exp_file_name(dict(wandb.config))
    model_name = get_model_name_from_config(wandb.config)

    ######################## DATASET ########################
    # Load dataset and split it if necessary
    train_set, test_set, collate_fn = load_dataset(wandb.config)

    # if there is pretraining, train the model on the prior set.
    if wandb.config['prior_size'] != 0.0:
        prior_set, train_set, validation_set = split_prior_train_validation_dataset(train_set, wandb.config['prior_size'], wandb.config['validation_size'])
    else:
        train_set, validation_set = split_train_validation_dataset(train_set, wandb.config['validation_size'])

    train_set, disagreement_set = split_train_validation_dataset(train_set, wandb.config['disagreement_size'])

    # We check everything is a CustomDataset and will work correctly
    assert isinstance(train_set, CustomDataset)
    assert isinstance(validation_set, CustomDataset)
    assert isinstance(test_set, CustomDataset)

    # Instantiate the mask that will deal with the indexes
    dataset_idx = CompressionSetIndexes(len(train_set))

    # create the dataloaders for the validation and test data. 
    trainset_loader = get_dataloader(dataset=train_set, batch_size=batch_size, collate_fn=collate_fn)
    valset_loader = get_dataloader(dataset=validation_set, batch_size=batch_size, collate_fn=collate_fn)
    disagreement_loader = get_dataloader(dataset=disagreement_set, batch_size=batch_size, collate_fn=collate_fn)
    test_loader = get_dataloader(dataset=test_set, batch_size=batch_size, collate_fn=collate_fn)

    ######################## MODEL ########################
    if wandb.config['prior_size'] == 0.0:
         model = create_model(wandb.config)
    else:
        raise NameError("This file does not support pretrained models.")
        
    # Updates the lr, as it might not be the same in the pretraining and training
    update_learning_rate(model, wandb.config.get('training_lr', None), len(trainset_loader))
    update_clamping_method(model, wandb.config['clamp_method'])
    if wandb.config['clamping']:
        add_clamping_to_model(model, config=wandb.config)
    
    # Forward pass of prediction to find on which data we do the most error
    prediction_trainer = get_trainer(accelerator=accelerator)
    validation_loss = log_metrics(prediction_trainer,
                model,
                trainset_loader,
                valset_loader,
                test_loader,
                0,
                len(train_set),
                n_sigma,
                return_validation_loss=True)
    errors = prediction_trainer.predict(model=model, dataloaders=trainset_loader)
    z, idx = get_max_error_idx(errors, wandb.config['data_groupsize'])
    
    # We need to correct the indices, as the continuously changing indices cause index shift
    idx = dataset_idx.correct_idx(idx)

    max_compression_size = len(train_set) if wandb.config['max_compression_size'] == -1 else wandb.config['max_compression_size']
    early_stopper = StoppingCriterion(max_compression_size,
                                    stop_criterion=STOP,
                                    patience=wandb.config['early_stopping_patience'],
                                    use_early_stopping=wandb.config['early_stopping'],
                                    use_p2l_stopping=not wandb.config['regression'])
    
    compression_set_size = dataset_idx.get_compression_size()
    early_stopper.check_stop(loss=validation_loss, max_error=z, compression_set_size=compression_set_size)

    # main loop of p2l
    while not early_stopper.stop:
        print(z.item(), compression_set_size)

        # update the compression set
        dataset_idx.update_compression_set(idx)
        compression_set_size = dataset_idx.get_compression_size()

        # train on the compression set
        compression_set = train_set.clone_dataset(dataset_idx.get_compression_data())
        compression_set.transform = get_data_augmentation_transform(wandb.config['dataset'])

        compression_loader = get_dataloader(dataset=compression_set,
                             batch_size=get_updated_batch_size(batch_size, wandb.config['model_type'], len(compression_set)),
                             collate_fn=collate_fn)
        trainer = get_trainer(accelerator=accelerator,
                            max_epochs=wandb.config['max_epochs'],
                            callbacks=[EarlyStopping(monitor="validation_loss", mode="min", patience=wandb.config['patience'])])

        trainer.fit(model=model, train_dataloaders=compression_loader, val_dataloaders=valset_loader)   

        # predict on the complement set
        complement_set = train_set.clone_dataset(dataset_idx.get_complement_data())
        complement_loader = get_dataloader(dataset=complement_set, batch_size=batch_size, collate_fn=collate_fn)
        errors = prediction_trainer.predict(model=model, dataloaders=complement_loader)
        z, idx = get_max_error_idx(errors, wandb.config['data_groupsize'])

        # We need to correct the indices, as the continuously changing indices cause index shift
        idx = dataset_idx.correct_idx(idx)
        
        wandb.log({'max_error': z})

        early_stopper.check_stop(loss=trainer.callback_metrics['validation_loss'],
                                max_error=z,
                                compression_set_size=compression_set_size)

        # On va tester le modèle sur le complement et validation set, ainsi que calculer les bornes
        if (compression_set_size ) % (wandb.config['data_groupsize'] * wandb.config['log_iterations']) == 0:
            log_metrics(prediction_trainer,
                        model,
                        complement_loader,
                        valset_loader,
                        test_loader,
                        len(compression_set),
                        len(train_set),
                        n_sigma)


    print(f"P2l ended with max error {z.item():.2f} and a compression set of size {compression_set_size}")

    # Test the model on the complement set
    complement_set = train_set.clone_dataset(dataset_idx.get_complement_data())
    complement_loader = get_dataloader(dataset=complement_set, batch_size=batch_size, collate_fn=collate_fn)
    complement_results = prediction_trainer.validate(model, dataloaders=complement_loader)

    # Test the model on the validation and test sets
    validation_results = prediction_trainer.validate(model, dataloaders=valset_loader)
    test_results = prediction_trainer.test(model, dataloaders=test_loader)

    # log informations
    if wandb.config['prior_size'] != 0.0:
        information_dict['prior_set_size'] = len(prior_set)
    information_dict['train_set_size'] = len(train_set)
    information_dict['validation_set_size'] = len(validation_set)
    information_dict['test_set_size'] = len(test_set)
    information_dict['compression_set_size'] = compression_set_size

    if not wandb.config['regression']:
        information_dict['complement_error'] = complement_results[0]['validation_error']
        information_dict['validation_error'] = validation_results[0]['validation_error']
        information_dict['test_error'] = test_results[0]['test_error']
    
    information_dict['complement_loss'] = complement_results[0]['validation_loss']
    information_dict['validation_loss'] = validation_results[0]['validation_loss']
    information_dict['test_loss'] = test_results[0]['test_loss']

    # compute the bounds
    n = len(train_set)
    
    if wandb.config['classic_bounds']:   
        print(("-"*20) + " Classical compression bounds " + "-"*20)
        k = int(complement_results[0]['validation_error'] * (n-compression_set_size))
        compute_classical_compression_bounds(compression_set_size, n_sigma, n, k, wandb.config['delta'], information_dict)

    if wandb.config['p2l_bounds']:
        print(("-"*20) + " Pick-to-learn bounds " + "-"*20)
        k = int(complement_results[0]['validation_error'] * (n-compression_set_size))
        compute_all_p2l_bounds(compression_set_size, n, k,  wandb.config['delta'], information_dict)
    
    if wandb.config['real_bounds']:
        print(("-"*20) + " Real valued bounds " + "-"*20)

        if wandb.config['regression']:
            compute_real_valued_bounds(compression_set_size, n_sigma, n, complement_results[0]['validation_loss'],
                                        wandb.config['delta'], wandb.config['nbr_parameter_bounds'], information_dict, 
                                        min_val=wandb.config['min_val'], max_val=wandb.config['max_val'])
        else:
            compute_real_valued_bounds(compression_set_size, n_sigma, n, complement_results[0]['validation_error'],
                                        wandb.config['delta'], wandb.config['nbr_parameter_bounds'], information_dict)
            if wandb.config['clamping']:
                print(("-"*20) + " Real valued bounds for bounded cross entropy" + "-"*20)
                min_val_loss, max_val_loss = get_min_max_loss(wandb.config['min_probability'], wandb.config['n_classes'], wandb.config['clamp_method'])
                compute_real_valued_bounds(compression_set_size, n_sigma, n, complement_results[0]['validation_loss'], wandb.config['delta'],
                                            wandb.config['nbr_parameter_bounds'], information_dict, min_val=min_val_loss, max_val=max_val_loss, prefix="CE")

    compute_disagreement(model, disagreement_loader, wandb.config, information_dict, device=device)
    wandb.log(information_dict)

    information_dict['config'] = dict(wandb.config)

    # save the experiment informations in a json
    if not os.path.isdir("./experiment_logs"):
        os.mkdir("./experiment_logs")

    with open(file_dir, "w") as outfile: 
        json.dump(information_dict, outfile)

    file_path = "./p2l_trained_models/"
    if not os.path.isdir(file_path):
        os.mkdir(file_path)

    file_path = file_path + model_name
    prediction_trainer.save_checkpoint(file_path)
    wandb.finish()

def hyperparameter_loop(list_of_sweep_configs, dataset_config):
    for sweep_config_ in list_of_sweep_configs:
        exp_config = sweep_config_ | dataset_config
        config_name = get_exp_file_name(exp_config)
        if not os.path.isfile(config_name):
            exp_name = sweep_configuration['name'] + dataset_config['dataset']
            if dataset_config.get('n_classes', -1) == 2 and dataset_config['dataset'] == "mnist":
                exp_name += str(dataset_config['first_class']) + str(dataset_config['second_class'])
            exp_name += "_ckpt"
            p2l_experiments(exp_config, name=exp_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default="mnist", help="Name of the dataset")
    args = parser.parse_args()

    sweep_config_name = "./configs/experiment_configs/p2l/" + args.dataset + ".yaml"
    with open(sweep_config_name) as file:
        sweep_configuration = yaml.safe_load(file)

    params_config_name = "./configs/dataset_configs/" + args.dataset + ".yaml"
    with open(params_config_name) as file:
        config = yaml.safe_load(file)

    # correct types of entries to make sure all floats/ints are parsed as such
    for key, value in config.items():
        config[key] = correct_type_of_entry(value)

    list_of_configs = create_all_configs(sweep_configuration)
    if config.get('regression', False) or config.get('n_classes', -1) != 2 or config['dataset'] != "mnist":
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