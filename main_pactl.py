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
import loralib as lora
from bounds.classical_bounds import compute_classical_compression_bounds
from bounds.bound_utils import zeta
from multiprocessing import Pool
from pytorch_lightning.loggers import WandbLogger
from copy import deepcopy

import sys
sys.path.insert(0, './Pactl')

from Pactl.pactl.nn.projectors import create_intrinsic_model, flatten, unflatten_like
from Pactl.pactl.bounds.get_bound_from_chk_v2 import compute_quantization
from Pactl.pactl.bounds.quantize_fns import quantize_vector

CUDA_DEVICE = 0
NUMBER_OF_SEED = 5

def pactl_experiments(config, name):
    wandb.init(project=name, config=config)
    logger = WandbLogger(project=name, experiment=wandb.run, prefix="(train)")
    file_name = get_exp_file_name(dict(wandb.config), path="./pactl_logs/")
    model_name = get_model_name_from_config(wandb.config, use_pactl=True)

    seed_everything(wandb.config['seed'], workers=True)
    accelerator = get_accelerator(wandb.config['model_type'])
    device = torch.device(f'cuda:{CUDA_DEVICE}' if torch.cuda.is_available() else 'cpu')

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

    transform = get_data_augmentation_transform(wandb.config['dataset'], "baseline")
    train_set.transform = transform
    if prior_set is not None:
        # prior_set.transform = transform
        prior_set.transform = get_data_augmentation_transform(wandb.config['dataset'], "p2l")

    if wandb.config.get('batch_size_val', None) is not None:
        test_batch_size = wandb.config['batch_size_val']
    trainset_loader = get_dataloader(dataset=train_set, batch_size=wandb.config['batch_size'], collate_fn=collate_fn)
    valset_loader = get_dataloader(dataset=validation_set, batch_size=wandb.config['batch_size'], collate_fn=collate_fn)
    test_loader = get_dataloader(dataset=test_set, batch_size=wandb.config['batch_size'], collate_fn=collate_fn)

    if wandb.config.get('distillation', False):
        best_model = get_best_model(wandb.config['dataset'], wandb.config['model_type'], wandb.config['seed'])
        create_distillation_targets(best_model, disagreement_set, device, collate_fn,  config)
        best_model = None
    disagreement_loader = get_dataloader(dataset=disagreement_set, batch_size=wandb.config['batch_size'], collate_fn=collate_fn)

    if wandb.config['prior_size'] == 0.0:
        model = create_model(wandb.config)
        if wandb.config['rank'] > 0:
            model.model = create_lora_model(wandb.config, model.model)
    else:
        file_path = "./prior_models/"
        if not os.path.isdir(file_path):
            os.mkdir(file_path)
        
        prior_model_name = f"{wandb.config['dataset']}_{wandb.config['model_type']}_{wandb.config['prior_size']}_{wandb.config['pretraining_lr']}_{wandb.config['pretraining_epochs']}_{wandb.config['seed']}.ckpt"
        file_path = file_path + prior_model_name
        if os.path.isfile(file_path):
            model = load_pretrained_model(file_path, wandb.config)
        else:
            prior_config = deepcopy(dict(wandb.config))
            prior_config['rank'] = 0
            model = create_model(prior_config)
            prior_loader= get_dataloader(dataset=prior_set, batch_size=wandb.config['batch_size'] , collate_fn=collate_fn)
            prior_trainer = get_trainer(accelerator=accelerator, max_epochs=wandb.config['pretraining_epochs'])
            prior_trainer.fit(model=model, train_dataloaders=prior_loader, val_dataloaders=trainset_loader)
            prior_trainer.save_checkpoint(file_path)

        if wandb.config['rank'] > 0:
            model.model = create_lora_model(wandb.config, model.model)

    if wandb.config['dataset'] == "amazon":
        for name, params in model.named_parameters():
            if "embedding" in name:
                params.requires_grad = False
            elif "layer" in name:
                if int(name[name.find("layer")+len("layer")+1]) < wandb.config['frozen_layer']:
                    params.requires_grad = False
    # add model
    if wandb.config['intrinsic_dim'] > 0:
        model.to(device)
        intrinsic_net = create_intrinsic_model(model.model, intrinsic_mode=wandb.config['intrinsic_mode'], intrinsic_dim=wandb.config['intrinsic_dim'], seed=wandb.config['seed'])
        model.model = intrinsic_net
        model.model.n_classes = wandb.config['n_classes']
    elif wandb.config['rank'] > 0:
        lora.mark_only_lora_as_trainable(model.model)

    update_learning_rate(model, wandb.config.get('training_lr', None), len(trainset_loader))
    update_clamping_method(model, wandb.config['clamp_method'])
    if wandb.config['clamping']:
        add_clamping_to_model(model, config=wandb.config)
    
    trainer = get_trainer(accelerator=accelerator,
                        max_epochs=wandb.config['max_epochs'], 
                        devices=[CUDA_DEVICE],
                        callbacks=[EarlyStopping(monitor="validation_error", patience=10)],
                            logger=logger)

    if wandb.config.get('distillation', False):
        concatenated_dataset = torch.utils.data.ConcatDataset([train_set, disagreement_set])
        loader_for_training = get_dataloader(dataset=concatenated_dataset, batch_size=wandb.config['batch_size'], collate_fn=collate_fn)
    else:
        loader_for_training = trainset_loader
    trainer.fit(model=model, train_dataloaders=loader_for_training, val_dataloaders=valset_loader)

    train_set.transform = validation_set.transform
    trainset_loader = get_dataloader(dataset=train_set, batch_size=wandb.config['batch_size'], collate_fn=collate_fn)

    # compute quantization
    
    if wandb.config['intrinsic_dim'] > 0:
        quant_model = model.model
        quant_model.to(device)
        quantized_vec, message_len = compute_quantization(quant_model,
                                                        levels=wandb.config['levels'],
                                                            device=device,
                                                            train_loader=trainset_loader,
                                                            epochs=wandb.config['quant_epochs'],
                                                            lr=1.0e-2,
                                                            use_kmeans=wandb.config['use_kmeans'])

        try:
            module = model.model
            if quantized_vec is not None:
                module.subspace_params.data = torch.tensor(quantized_vec).float().to(device)
            else:
                aux = torch.zeros_like(module.subspace_params.data).float().to(device)
                module.subspace_params.data = aux
        except AttributeError:
            print("Quantization vector was not updated.")
    else:
        aux = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        names, vector = zip(*aux)
        fvector = flatten(vector).cpu().data.numpy()
        quantized_vec, message_len = quantize_vector(fvector,
                                                    levels=wandb.config['levels'],
                                                    use_kmeans=wandb.config['use_kmeans'])
        ## free memory 
        fvector = None 
        unfquantized_vec = unflatten_like(torch.tensor(quantized_vec), vector)
        ## free memory  
        quantized_vec, vector = None, None
        for n, p in model.named_parameters():
            for name, quantp in zip(names, unfquantized_vec):
                if n == name:
                    p.data = torch.tensor(quantp).float().to(device)

    file_path = "./pactl_trained_models/"
    if not os.path.isdir(file_path):
        os.mkdir(file_path)
    file_path = file_path + model_name
    trainer.save_checkpoint(file_path)

    prefix_message_len = message_len + 2 * np.log2(message_len) if message_len > 0 else 0
    misc_extra_bits = np.log2(1) # number of hyperparameters (we consider none right now)
    divergence = (prefix_message_len + misc_extra_bits) * np.log(2) # bound on log(1/P(h))

    train_results = trainer.validate(model=model, dataloaders=trainset_loader)
    validation_results = trainer.validate(model=model, dataloaders=valset_loader)
    test_results = trainer.test(model, dataloaders=test_loader)
    
    logger._prefix = ""
    information_dict = {}

    print("This is the message length : ", message_len)
    print("This is the complement error : ", train_results[0]['validation_error'])
    
    information_dict['train_set_size'] = len(train_set)
    information_dict['validation_set_size'] = len(validation_set)
    information_dict['test_set_size'] = len(test_set)

    information_dict['complement_error'] = train_results[0]['validation_error']
    information_dict['validation_error'] = validation_results[0]['validation_error']
    information_dict['test_error'] = test_results[0]['test_error']

    information_dict['complement_loss'] = train_results[0]['validation_loss']
    information_dict['validation_loss'] = validation_results[0]['validation_loss']
    information_dict['test_loss'] = test_results[0]['test_loss']

    if train_results[0].get('validation_huber_loss', None) is not None:
        information_dict['complement_huber_loss'] = train_results[0]['validation_huber_loss']
        information_dict['validation_huber_loss'] = validation_results[0]['validation_huber_loss']
        information_dict['test_huber_loss'] = test_results[0]['test_huber_loss']

    information_dict['message_len'] = message_len

    # compute the bound
    n = len(train_set)
    compression_set_size = 0
    n_sigma = zeta(compression_set_size) # correction term when there is no compression set
    if wandb.config['classic_bounds']:   
        print(("-"*20) + " Classical compression bounds " + "-"*20)
        k = int(train_results[0]['validation_error'] * n)
        
        compute_classical_compression_bounds(compression_set_size, n_sigma, n, k, wandb.config['delta'], information_dict, log_divergence=divergence)
    
    # compute the bound with the divergence
    if wandb.config['real_bounds']:
        print(("-"*20) + " Real valued bounds " + "-"*20)

        compute_real_valued_bounds(compression_set_size, n_sigma, n, train_results[0]['validation_error'],
                                    wandb.config['delta'], wandb.config['nbr_parameter_bounds'], information_dict, log_divergence=divergence)
        
        print(("-"*20) + " Real valued bounds for bounded cross entropy" + "-"*20)
        min_val_loss, max_val_loss = get_min_max_loss(wandb.config['min_probability'], wandb.config['n_classes'], 
                                                      wandb.config['clamp_method'], wandb.config.get('huber_delta', 0.2))
        if wandb.config['clamp_method'] is None:
            loss_for_bound = train_results[0]['validation_huber_loss']
            prefix = "huber"
        else:
            loss_for_bound = train_results[0]['validation_loss']
            prefix = "CE"
        compute_real_valued_bounds(compression_set_size, n_sigma, n, loss_for_bound, wandb.config['delta'],
                                    wandb.config['nbr_parameter_bounds'], information_dict, min_val=min_val_loss,
                                    max_val=max_val_loss, prefix=prefix, log_divergence=divergence)
        
    log_divergence = divergence if wandb.config.get('distillation', False) else None
    compute_disagreement(model, disagreement_loader, wandb.config, information_dict, device, log_divergence=log_divergence)

    wandb.log(information_dict)
    information_dict['config'] = dict(wandb.config)

    # save the experiment informations in a json
    if not os.path.isdir("./pactl_logs"):
        os.mkdir("./pactl_logs")

    with open(file_name, "w") as outfile: 
        json.dump(information_dict, outfile)

    wandb.finish()

    raise NotImplementedError("Donezo!")

def hyperparameter_loop(list_of_sweep_configs, dataset_config):
    for sweep_config_ in list_of_sweep_configs:
        exp_config = sweep_config_ | dataset_config
        config_name = get_exp_file_name(exp_config, path="./pactl_logs/")
        if not os.path.isfile(config_name):
            exp_name = 'pactl_' + dataset_config['dataset'] + "_ckpt"
            if dataset_config.get('n_classes', -1) == 2 and dataset_config['dataset'] == "mnist":
                exp_name += str(dataset_config['first_class']) + str(dataset_config['second_class'])
            pactl_experiments(exp_config, name=exp_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default="mnist", help="Name of the dataset")
    args = parser.parse_args()

    sweep_config_name = "./configs/experiment_configs/pactl/" + args.dataset + ".yaml"
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
