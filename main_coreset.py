import torch
from lightning.pytorch import seed_everything
import lightning as L
from utilities.utils import *
from utilities.utils_compression_set import *
from utilities.utils_datasets import *
from utilities.utils_early_stopping import *
from utilities.utils_models import *
from dataset.dataset_loader import load_dataset
from bounds.classical_bounds import compute_classical_compression_bounds,\
    brute_force_binomial_tail_inversion, binomial_approximation
from bounds.real_valued_bounds import compute_real_valued_bounds
from bounds.kl_inv import kl_inv
import os
import json
import argparse
import wandb
import yaml
from copy import deepcopy
from pytorch_lightning.loggers import WandbLogger
import DeepCore.deepcore.methods as methods

CUDA_DEVICE = 0


def coreset_experiments(config, name):
    torch.cuda.empty_cache()
    wandb.init(project=name, config=config)
    logger = WandbLogger(project=name, experiment=wandb.run, prefix="(train)")
    file_name = get_exp_file_name(dict(wandb.config), path="./coreset_logs/")
    model_name = get_model_name_from_config(wandb.config)

    n_sigma = 1

    seed_everything(wandb.config['seed'], workers=True)
    accelerator = get_accelerator(wandb.config['model_type'])
    device = torch.device(f'cuda:{CUDA_DEVICE}' if torch.cuda.is_available() else 'cpu')

    # create models, load dataset and split it if necessary
    
    train_set, test_set, collate_fn = load_dataset(wandb.config)
    if wandb.config['prior_size'] != 0.0:
        prior_set, train_set, validation_set = split_prior_train_validation_dataset(train_set, wandb.config['prior_size'], wandb.config['validation_size'])
    else:
        train_set, validation_set = split_train_validation_dataset(train_set, wandb.config['validation_size'])
        prior_set = None

    train_set, disagreement_set = split_train_validation_dataset(train_set, wandb.config.get('disagreement_size',0.2))

    indice_file_path = "./indices_coresets/"
    if not os.path.isdir(indice_file_path):
        os.mkdir(indice_file_path)
    if wandb.config['optimizer'] == "sgdfree":
        optimizer = "SGD"
    elif wandb.config['optimizer'] == "adamfree":
        optimizer = "Adam"
    else:
        optimizer = wandb.config['optimizer']
    indices_file_name = f"{wandb.config['dataset']}_{wandb.config['seed']}_{wandb.config['model_type']}_{wandb.config['selection_epochs']}_{wandb.config['selection']}"\
        +f"_{wandb.config['fraction']}_{wandb.config['submodular']}_{wandb.config['submodular_greedy']}_{wandb.config['uncertainty']}_{optimizer}_"\
        +f"{wandb.config['training_lr']}_{wandb.config['weight_decay']}.npy"
    
    indices_file_name = indice_file_path + indices_file_name

    if os.path.isfile(indices_file_name):
        indices = np.load(indices_file_name)
    else:
        # Subset selection with DeepCore
        args = argparse.Namespace(**dict(wandb.config))
        args.channel = train_set[0][0].shape[0]
        args.im_size = [train_set[0][0].shape[1]]*2
        args.num_classes = wandb.config['n_classes']
        args.print_freq = 20
        args.selection_batch = wandb.config['batch_size']
        args.selection_momentum = wandb.config['momentum']
        args.selection_weight_decay = wandb.config['weight_decay']
        args.selection_optimizer = optimizer
        args.selection_nesterov = wandb.config['nesterov']
        args.selection_lr = wandb.config['training_lr']
        args.balance = False
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        args.gpu = [0]
        args.workers = 4

        train_set.classes = np.arange(args.num_classes)

        if wandb.config['dataset'] == "mnist":
            args.model = "MnistCnnAdapter"
        elif wandb.config['dataset'] == 'cifar10':
            if wandb.config['model_type'] == "cnn":
                args.model = "Cifar10CnnAdapter"
            elif wandb.config['model_type'] == "resnet":
                args.model = "ResNet18"

        selection_args = dict(epochs=args.selection_epochs,
                                selection_method=args.uncertainty,
                                balance=args.balance,
                                greedy=args.submodular_greedy,
                                function=args.submodular
                                )
        if args.selection == 'kCenterGreedy':
            selection_args['specific_model'] = 'Resnet18Adapter'
        if args.selection == 'Cal':
            selection_args['pretrain_model'] = 'Resnet18Adapter'
        method = methods.__dict__[args.selection](train_set, args, args.fraction, args.seed, **selection_args)
        subset = method.select()
        indices = subset["indices"].copy()
        np.save(indices_file_name, indices)

    dataset_idx = CompressionSetIndexes(len(train_set))
    dataset_idx.update_compression_set(indices) 
    
    assert dataset_idx.get_compression_size() == len(indices) # save the indices

    compression_set = train_set.clone_dataset(dataset_idx.get_compression_data())
    complement_set = train_set.clone_dataset(dataset_idx.get_complement_data())

    # Training of the model
    transform = get_data_augmentation_transform(wandb.config['dataset'])
    compression_set.transform = transform
    if prior_set is not None:
        prior_set.transform = transform

    compression_set_loader = get_dataloader(dataset=compression_set, batch_size=wandb.config['batch_size'], collate_fn=collate_fn)
    complement_set_loader = get_dataloader(dataset=complement_set, batch_size=wandb.config['batch_size'], collate_fn=collate_fn)

    valset_loader = get_dataloader(dataset=validation_set, batch_size=wandb.config['batch_size'], collate_fn=collate_fn)
    disagreement_loader = get_dataloader(dataset=disagreement_set, batch_size=wandb.config['batch_size'], collate_fn=collate_fn)
    test_loader = get_dataloader(dataset=test_set, batch_size=wandb.config['batch_size'], collate_fn=collate_fn)


    if wandb.config['prior_size'] == 0.0:
        model = create_model(wandb.config)
    else:
        raise NameError("This file does not support pretrained models.")

    update_learning_rate(model, wandb.config.get('training_lr', None), len(compression_set_loader))
    update_clamping_method(model, wandb.config['clamp_method'])
    if wandb.config['clamping']:
        add_clamping_to_model(model, config=wandb.config)
    
    trainer = get_trainer(accelerator=accelerator,
                        max_epochs=wandb.config['max_epochs'], 
                        callbacks=LearningRateMonitor(logging_interval="step"),
                            logger=logger)
    trainer.fit(model=model, train_dataloaders=compression_set_loader, val_dataloaders=valset_loader)

    file_path = "./coreset_trained_models/"
    if not os.path.isdir(file_path):
        os.mkdir(file_path)
    file_path = file_path + model_name
    trainer.save_checkpoint(file_path)

    complement_results = trainer.validate(model=model, dataloaders=complement_set_loader)
    validation_results = trainer.validate(model=model, dataloaders=valset_loader)
    test_results = trainer.test(model, dataloaders=test_loader)
    
    logger._prefix = ""
    information_dict = {}

    information_dict['train_set_size'] = len(train_set)
    information_dict['compression_set_size'] = len(compression_set)
    information_dict['validation_set_size'] = len(validation_set)
    information_dict['test_set_size'] = len(test_set)

    information_dict['complement_error'] = complement_results[0]['validation_error']
    information_dict['validation_error'] = validation_results[0]['validation_error']
    information_dict['test_error'] = test_results[0]['test_error']

    information_dict['complement_loss'] = complement_results[0]['validation_loss']
    information_dict['validation_loss'] = validation_results[0]['validation_loss']
    information_dict['test_loss'] = test_results[0]['test_loss']

    compression_set_size = dataset_idx.get_compression_size()
    n = len(train_set)

    if wandb.config['classic_bounds']:   
        print(("-"*20) + " Classical compression bounds " + "-"*20)
        k = int(complement_results[0]['validation_error'] * (n-compression_set_size))
        compute_classical_compression_bounds(compression_set_size, n_sigma, n, k, wandb.config['delta'], information_dict)

    if wandb.config['real_bounds']:
        print(("-"*20) + " Real valued bounds " + "-"*20)

        compute_real_valued_bounds(compression_set_size, n_sigma, n, complement_results[0]['validation_error'],
                                    wandb.config['delta'], wandb.config['nbr_parameter_bounds'], information_dict)
        if wandb.config['clamping']:
            print(("-"*20) + " Real valued bounds for bounded cross entropy" + "-"*20)
            min_val_loss, max_val_loss = get_min_max_loss(wandb.config['min_probability'], wandb.config['n_classes'], wandb.config['clamp_method'])
            compute_real_valued_bounds(compression_set_size, n_sigma, n, complement_results[0]['validation_loss'], wandb.config['delta'],
                                        wandb.config['nbr_parameter_bounds'], information_dict, min_val=min_val_loss, max_val=max_val_loss, prefix="CE")

    if wandb.config['selection'] == "Uniform":
        k = int(complement_results[0]['validation_error'] * len(complement_set))
        information_dict['test_set_bound_brute'] = brute_force_binomial_tail_inversion(k, len(complement_set), 
                                                                                       np.log(wandb.config['delta']))
        information_dict['test_set_bound_binomial_approx'] = binomial_approximation(k, len(complement_set),
                                                                                    np.log(wandb.config['delta']))
        if wandb.config['clamping']:
            min_val_loss, max_val_loss = get_min_max_loss(wandb.config['min_probability'],
                                                           wandb.config['n_classes'],
                                                             wandb.config['clamp_method'])
            lambda_loss = max_val_loss - min_val_loss
            normalized_loss = (complement_results[0]['validation_loss'] - min_val_loss) / lambda_loss
            kl_bound = kl_inv(normalized_loss, np.log(1/wandb.config['delta'])/len(complement_set), "MAX")
            information_dict['test_set_CE_bound'] = min_val_loss + lambda_loss * kl_bound

    compute_disagreement(model, disagreement_loader, wandb.config, information_dict, device=device)
    
    wandb.log(information_dict)
    information_dict['config'] = dict(wandb.config)

    # save the experiment informations in a json
    if not os.path.isdir("./coreset_logs"):
        os.mkdir("./coreset_logs")

    with open(file_name, "w") as outfile: 
        json.dump(information_dict, outfile)

    wandb.finish()

def hyperparameter_loop(list_of_sweep_configs, dataset_config):
    for sweep_config_ in list_of_sweep_configs:
        exp_config = sweep_config_ | dataset_config
        config_name = get_exp_file_name(exp_config, path="./coreset_logs/")
        if not os.path.isfile(config_name):
            exp_name = 'coreset_' + dataset_config['dataset'] + "_ckpt"
            if dataset_config.get('n_classes', -1) == 2 and dataset_config['dataset'] == "mnist":
                exp_name += str(dataset_config['first_class']) + str(dataset_config['second_class'])
            coreset_experiments(exp_config, name=exp_name)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default="mnist", help="Name of the dataset")
    args = parser.parse_args()

    sweep_config_name = "./configs/experiment_configs/coreset/" + args.dataset + ".yaml"
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
