import torch
from lightning.pytorch import seed_everything
from utilities.utils import *
from utilities.utils_compression_set import *
from utilities.utils_datasets import *
from utilities.utils_early_stopping import *
from utilities.utils_models import *
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from dataset.dataset_loader import load_dataset
import os
import json
import argparse
import wandb
import yaml
from copy import deepcopy
from pytorch_lightning.loggers import WandbLogger

class CustomCallback(EarlyStopping):
    def __init__(self, monitor, stop):
        super().__init__(monitor=monitor)
        self.monitor = monitor
        self.stop = stop
        self.mode = "="
        self.mode_dict = {"=": torch.eq}

    def _evaluate_stopping_criteria(self, current):
        should_stop = self.monitor_op(current, self.stop)
        return should_stop, None

def baseline(config, name):
    wandb.init(project=name, config=config)
    logger = WandbLogger(project=name, experiment=wandb.run, prefix="(train)")
    file_name = get_exp_file_name(dict(wandb.config), path="./baseline_logs/")
    model_name = get_model_name_from_config(wandb.config)

    seed_everything(wandb.config['seed'], workers=True)
    accelerator = get_accelerator(wandb.config['model_type'])

    # create models, load dataset and split it if necessary
    
    train_set, test_set, collate_fn = load_dataset(wandb.config)
    if wandb.config['prior_size'] != 0.0:
        prior_set, train_set, validation_set = split_prior_train_validation_dataset(train_set, wandb.config['prior_size'], wandb.config['validation_size'])
        raise NotImplementedError("I did not implement yet the prior set with respect to the disagreement set.")
    else:
        train_set, validation_set = split_train_validation_dataset(train_set, wandb.config['validation_size'])
        prior_set = None

    train_set, disagreement_set = split_train_validation_dataset(train_set, wandb.config.get('disagreement_size',0.2))

    transform = get_data_augmentation_transform(wandb.config['dataset'], "baseline")
    train_set.transform = transform
    if prior_set is not None:
        prior_set.transform = transform

    trainset_loader = get_dataloader(dataset=train_set, batch_size=wandb.config['batch_size'], collate_fn=collate_fn)
    valset_loader = get_dataloader(dataset=validation_set, batch_size=wandb.config['batch_size'], collate_fn=collate_fn)
    test_loader = get_dataloader(dataset=test_set, batch_size=wandb.config['batch_size'], collate_fn=collate_fn)

    model = create_model(wandb.config)

    for name, params in model.named_parameters():
        if "embedding" in name:
            params.requires_grad = False
    
    update_learning_rate(model, wandb.config.get('training_lr', None), len(trainset_loader))
    update_clamping_method(model, wandb.config['clamp_method'])
    if wandb.config['clamping']:
        add_clamping_to_model(model, config=wandb.config)
    
    trainer = get_trainer(accelerator=accelerator,
                        max_epochs=wandb.config['max_epochs'], 
                        #   callbacks=[CustomCallback(monitor="validation_error", stop=0.0)],
                        callbacks=get_dataset_callbacks(wandb.config['dataset']),
                            logger=logger)
    trainer.fit(model=model, train_dataloaders=trainset_loader, val_dataloaders=valset_loader)

    file_path = "./trained_models/"
    if not os.path.isdir(file_path):
        os.mkdir(file_path)
    file_path = file_path + model_name
    trainer.save_checkpoint(file_path, weights_only=True)

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

    wandb.log(information_dict)
    information_dict['config'] = dict(wandb.config)

    # save the experiment informations in a json
    if not os.path.isdir("./baseline_logs"):
        os.mkdir("./baseline_logs")

    with open(file_name, "w") as outfile: 
        json.dump(information_dict, outfile)

    wandb.finish()

def hyperparameter_loop(list_of_sweep_configs, dataset_config):
    for sweep_config_ in list_of_sweep_configs:
        exp_config = sweep_config_ | dataset_config
        config_name = get_exp_file_name(exp_config, path="./baseline_logs/")
        if not os.path.isfile(config_name):
            exp_name = 'baseline_' + dataset_config['dataset'] + "_ckpt"
            if dataset_config.get('n_classes', -1) == 2 and dataset_config['dataset'] == "mnist":
                exp_name += str(dataset_config['first_class']) + str(dataset_config['second_class'])
            baseline(exp_config, name=exp_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default="amazon", help="Name of the dataset")
    args = parser.parse_args()

    sweep_config_name = "./configs/dataset_configs/baseline/" + args.dataset + ".yaml"
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