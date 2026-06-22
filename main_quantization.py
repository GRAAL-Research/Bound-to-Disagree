import torch
from lightning.pytorch import seed_everything
import lightning as L
import transformers
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
from bounds.classical_bounds import compute_classical_compression_bounds
from bounds.bound_utils import zeta
from multiprocessing import Pool
from pytorch_lightning.loggers import WandbLogger
from copy import deepcopy
from torchao.quantization import Int4WeightOnlyConfig, quantize_
from torchao.quantization.qat import QATConfig
from torchao.dtypes import SemiSparseLayout
import subprocess
from hqq.core.quantize import *

CUDA_DEVICE = 0
NUMBER_OF_SEED = 5

def pactl_experiments(config, name):
    wandb.init(project=name, config=config)
    logger = WandbLogger(project=name, experiment=wandb.run, prefix="(train)")
    file_name = get_exp_file_name(dict(wandb.config), path="./pactl_logs/")
    model_name = get_model_name_from_config(wandb.config, use_pactl=True)

    seed_everything(wandb.config['seed'], workers=True)
    accelerator = get_accelerator(wandb.config['model_type'])
    torch.set_float32_matmul_precision('medium')
    # create models, load dataset and split it if necessary

    device = torch.device(f'cuda:{CUDA_DEVICE}' if torch.cuda.is_available() else 'cpu')
    
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
    valset_loader = get_dataloader(dataset=validation_set, batch_size=test_batch_size, collate_fn=collate_fn)
    test_loader = get_dataloader(dataset=test_set, batch_size=test_batch_size, collate_fn=collate_fn)
    disagreement_loader = get_dataloader(dataset=disagreement_set, batch_size=test_batch_size, collate_fn=collate_fn,
                                          num_workers=0, persistent_workers=False)

    model = get_best_model(wandb.config['dataset'], wandb.config['model_type'], wandb.config['seed'], number_of_seeds=NUMBER_OF_SEED)
    if wandb.config.get('distillation', False):
        create_distillation_targets(model, disagreement_set, device, collate_fn,  config)
    
    ## torch.ao pruning :
    if wandb.config['pruning_factor'] is not None:
        if wandb.config['nbits'] != 8:
            raise NotImplementedError(f"Pruning with {wandb.config['nbits']} is not implemented.")

        parameters_to_prune = []
        for name, module in model.named_modules():
            if "embeddings" in name:
                continue

            if hasattr(module, 'weight') and module.weight is not None:
                if "layer" not in name:
                    parameters_to_prune.append((module, 'weight'))
                elif "layer" in name and int(name[name.find("layer")+len("layer")+1]) >= wandb.config['frozen_layer']:
                    parameters_to_prune.append((module, 'weight'))

        torch.nn.utils.prune.global_unstructured(
            parameters_to_prune,
            pruning_method=torch.nn.utils.prune.L1Unstructured,
            amount=wandb.config['pruning_factor'],
            )

        for m in model.modules():
            if hasattr(m, 'weight') and torch.nn.utils.prune.is_pruned(m):
                torch.nn.utils.prune.remove(m, 'weight') 

    update_learning_rate(model, wandb.config.get('training_lr', None), len(trainset_loader))
    update_clamping_method(model, wandb.config['clamp_method'])
    if wandb.config['clamping']:
        add_clamping_to_model(model, config=wandb.config)

    trainer = get_trainer(accelerator=accelerator,
        max_epochs=wandb.config['max_epochs'], 
        devices=[CUDA_DEVICE],
        callbacks=[EarlyStopping(monitor="validation_error", patience=2)],
            logger=logger)
    if wandb.config['qat']:
        if wandb.config['nbits'] == 8:
            model.qconfig = torch.quantization.get_default_qat_qconfig()
            for _, mod in model.named_modules():
                if isinstance(mod, torch.nn.Embedding) or isinstance(mod, torch.nn.LayerNorm):
                    mod.qconfig = None

            model.train()
            torch.ao.quantization.prepare_qat(model, inplace=True)
        elif wandb.config['nbits'] == 4:
            base_config = Int4WeightOnlyConfig(group_size=32, layout=SemiSparseLayout)
            quantize_(model, QATConfig(base_config, step="prepare"))

        for name, params in model.named_parameters():
            if "embedding" in name:
                params.requires_grad = False
            elif "layer" in name and int(name[name.find("layer")+len("layer")+1]) < wandb.config['frozen_layer']:
                params.requires_grad = False
        torch.compile(model)
        
        if wandb.config.get('distillation', False):
            concatenated_dataset = torch.utils.data.ConcatDataset([train_set, disagreement_set])
            loader_for_training = get_dataloader(dataset=concatenated_dataset, batch_size=wandb.config['batch_size'], collate_fn=collate_fn)
        else:
            loader_for_training = trainset_loader
        trainer.fit(model=model, train_dataloaders=loader_for_training, val_dataloaders=valset_loader)
    
    train_set.transform = validation_set.transform
    trainset_loader = get_dataloader(dataset=train_set, batch_size=wandb.config['batch_size'], collate_fn=collate_fn)

    quantized_model = None
    if wandb.config['qat']:
        if wandb.config['nbits'] == 8:
            quantized_model = torch.ao.quantization.convert(model, remove_qconfig=True)
        elif wandb.config['nbits'] == 4:
            quantized_model = deepcopy(model)
            quantize_(quantized_model, QATConfig(base_config, step="convert"))
    else:
        quant_config = BaseQuantizeConfig(nbits=wandb.config['nbits'], group_size=64)
        for name, module in model.named_modules():
            if (isinstance(module, torch.nn.Linear) or isinstance(module, transformers.pytorch_utils.Conv1D))\
                    and name not in ["model.model.score", "model.model.classifier"]:
                if isinstance(module, transformers.pytorch_utils.Conv1D):
                    with torch.no_grad():
                        lin_layer = torch.nn.Linear(in_features=module.weight.shape[0],
                                                    out_features=module.weight.shape[1],
                                                    bias=module.bias is not None)
                        lin_layer.weight = torch.nn.Parameter(module.weight.T)
                        lin_layer.bias = module.bias
                else:
                    lin_layer = deepcopy(module)
                temp = HQQLinear(lin_layer, #torch.nn.Linear or None 
                        quant_config=quant_config, #quantization configuration
                        compute_dtype=torch.float32, #compute dtype
                        device=device, #cuda device
                        initialize=True, #Use False to quantize later
                        del_orig=True #if True, delete the original layer
                        )
                new_name = name[name.find(".")+1:]
                mod = getattr(model, name[:name.find(".")])
                while "." in new_name:
                    mod = getattr(mod, new_name[:new_name.find(".")])
                    new_name = new_name[new_name.find(".")+1:]
                setattr(mod, new_name, temp)

    if quantized_model is not None:
        my_state_dict = quantized_model.model.model.state_dict()
        quantized_model = None # remove copy of model
    else:
        my_state_dict = model.model.model.state_dict()

    new_state_dict = {}
    ignore_vals = ["embeddings", "nbits", "group_size", "axis", "view_as_float", 
                   "encoded_state_dict", "stores_quant_config", "channel_wise", 
                   "optimize", "round_zero", "shape"]
    
    for k, v in my_state_dict.items():
        ignore_ = False
        for val in ignore_vals:
            if val in k:
                ignore_ = True

        if ignore_:
            continue
        elif "layer" in k and int(k[k.find("layer")+len("layer")+1]) < wandb.config['frozen_layer']:
            continue
        new_state_dict[k] = v

    file_path = "./pactl_trained_models/"
    if not os.path.isdir(file_path):
        os.mkdir(file_path)
    file_path = file_path + model_name
    try:
        trainer.save_checkpoint(file_path)
    except AttributeError:
        ...

    state_dict_path = file_path + ".pth"
    compressed_state_dict_path = state_dict_path + ".xz"

    torch.save(new_state_dict, state_dict_path)
    
    # with open(compressed_state_dict_path, "wb") as f_out:
    #      subprocess.run(["gzip", "-c", state_dict_path], stdout=f_out, check=True)
    with open(compressed_state_dict_path, "wb") as f_out:
        subprocess.run(["lzma", "-zc", state_dict_path], stdout=f_out, check=True)

    os.remove(state_dict_path)

    message_len = 8 * os.path.getsize(compressed_state_dict_path)

    prefix_message_len = message_len + 2 * np.log2(message_len) if message_len > 0 else 0
    misc_extra_bits = np.log2(1) # number of hyperparameters (we consider none right now)
    divergence = (prefix_message_len + misc_extra_bits) * np.log(2) # bound on log(1/P(h))
    
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
        
    if wandb.config.get('distillation', False):
        log_divergence = divergence
    else:
        log_divergence = None
    compute_disagreement(model, disagreement_loader, wandb.config, information_dict, device, log_divergence=log_divergence)

    wandb.log(information_dict)
    information_dict['config'] = dict(wandb.config)

    # save the experiment informations in a json
    if not os.path.isdir("./pactl_logs"):
        os.mkdir("./pactl_logs")

    with open(file_name, "w") as outfile: 
        json.dump(information_dict, outfile)

    wandb.finish()

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
    parser.add_argument('-d', '--dataset', type=str, default="amazon", help="Name of the dataset")
    args = parser.parse_args()

    sweep_config_name = "./configs/experiment_configs/quantize/" + args.dataset + ".yaml"
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
