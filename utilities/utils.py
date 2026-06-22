import torch
import lightning as L
from itertools import product
import wandb
from bounds.real_valued_bounds import compute_real_valued_bounds
from bounds.p2l_bounds import compute_all_p2l_bounds
from bounds.disagreement_bounds import compute_disagreement_bounds
import numpy as np
from models.classification_model import ClampedCrossEntropyLoss, SmoothedCrossEntropyLoss, ClippedCrossEntropyLoss
from utilities.utils_models import load_pretrained_model
import os
import json
from tqdm import tqdm
import operator
import transformers

def get_max_error_idx(errors, k):
    error_tensor = torch.cat(errors)
    # on gère le cas où il reste moins de k données dans le jeu de données.
    if error_tensor.shape[0] < k:
        k = error_tensor.shape[0]
    values, indices = torch.topk(error_tensor, k)
    return values.max(), indices

def update_learning_rate(model, lr:float, nb_batches=None) -> None:
    model.lr = lr
    model.nb_batches = nb_batches

def update_clamping_method(model, clamp_method):
    model.clamp_method = clamp_method

def add_clamping_to_model(model, config) -> None:
    if config['regression']:
        model.configure_loss(clamping=True, min_val=config['min_val'], max_val=config['max_val'])
    else:
        model.configure_loss(clamping=True, pmin=config['min_probability'])

def check_can_be_converted_to_float(entry) -> bool:
    try:
        float_entry = float(entry)
        return True
    except ValueError:
        return False

def correct_type_of_entry(entry):
    if isinstance(entry, list):
        return [correct_type_of_entry(entry_) for entry_ in entry]
    elif isinstance(entry, float):
        return entry
    elif isinstance(entry, str):
        if entry == 'None':
            return None
        elif check_can_be_converted_to_float(entry):
            return float(entry)
        else:
            return entry
    elif isinstance(entry, int):
        return entry
    else:
        raise ValueError(f'The entry type of {entry} is not recognised.')


def create_all_configs(config):
    if config['method'] != 'grid':
        raise NotImplementedError(f'The hyperparameter tuning method {config['method']} is not supported.')
    
    list_of_keys = []
    list_of_hyperparams = []
    if_statements = []
    replace_statements = []

    for key, item in config['parameters'].items():
        list_of_keys.append(key)
        if item.get('values', None) is not None:
            val_ = correct_type_of_entry(item['values'])
            list_of_hyperparams.append(val_)
            if item.get("if", None) is not None:
                if_statements.append((key, item['if']))
        elif item.get('value', None) is not None:
            val_ = correct_type_of_entry(item['value'])
            list_of_hyperparams.append([val_])
            if item.get("if", None) is not None:
                replace_statements.append((key, item['if']))
        else:
            raise ValueError(f"The parameter {key} doesn't have an item 'value' or 'values'. Please specify one.")
    
    list_of_configs = list(product(*list_of_hyperparams))
    results = [dict(zip(list_of_keys, config_)) for config_ in list_of_configs]

    if len(if_statements) == 0 and len(replace_statements) == 0:
        return results
    
    op_dict = {"==":operator.eq, "!=":operator.ne, "in":operator.contains}
    valid_configs = []
    for config in results:
        is_valid = True
        for key, if_ in if_statements:
            default_val = correct_type_of_entry(if_['default'])
            if config[key] == default_val:
                continue
            val = correct_type_of_entry(if_['val'])
            
            if not op_dict[if_['op']](val, config[if_['elem']]):
                is_valid = False
                break
        if is_valid:
            valid_configs.append(config)

    for config in valid_configs:
        for key, if_ in replace_statements:
            else_ = correct_type_of_entry(if_['else'])
            val = correct_type_of_entry(if_['val'])
            if not op_dict[if_['op']](val, config[if_['elem']]):
                config[key] = else_
    return valid_configs

def get_exp_file_name(config, path="./experiment_logs/"):
    list_of_params = list(config.values())
    file_name = "exp_"
    for param in list_of_params:
        file_name += str(param) + "_"
    file_name += ".json"
    return path + file_name

def get_updated_batch_size(batch_size, model_type, dataset_length):
    """
    When batch_size == -1, we want to train on the whole dataset.
    """
    if model_type in ['tree', 'forest']:
        return dataset_length
    return batch_size

def get_dataloader(dataset, batch_size, shuffle=False, num_workers=5, persistent_workers=True, collate_fn=None):
    return torch.utils.data.DataLoader(dataset,
                                        batch_size=batch_size,
                                        shuffle=shuffle,
                                        num_workers=num_workers, 
                                        persistent_workers=persistent_workers, 
                                        collate_fn=collate_fn)

def get_trainer(accelerator='auto', devices=1, max_epochs=None, logger=False,
                enable_checkpointing=False, callbacks=None):
    return L.Trainer(accelerator=accelerator,
                     devices=devices,
                    max_epochs=max_epochs,
                    logger=logger,
                    enable_checkpointing=enable_checkpointing,
                    callbacks=callbacks)

def get_accelerator(model_type:str):
    return 'cpu' if model_type in ["tree", 'forest'] else 'auto'

def get_min_max_loss(min_probability, n_classes, clamp_method, huber_delta=None):
    if clamp_method == "clamp":
        min_val = 0.0
        max_val = -np.log(min_probability/n_classes)
        assert min_val < max_val
        return min_val, max_val
    elif clamp_method == "smooth":
        max_val = -np.log(min_probability/n_classes)
        min_val = max_val - np.log(1+((1-min_probability)*n_classes/min_probability))
        assert min_val < max_val
        return min_val, max_val
    elif clamp_method == "clip":
        max_val = np.log(1+(n_classes-1)*np.exp(min_probability * np.sqrt(n_classes/(n_classes-1)))) 
        min_val = 0.0
        assert min_val < max_val
        return min_val, max_val
    elif clamp_method is None:
        # We use min_probability instead of delta for the huber loss
        min_val = 0.0
        max_val = huber_delta - (huber_delta **2) / 2
        assert min_val < max_val
        return min_val, max_val
    else:
        raise ValueError(f"Clamping method {clamp_method} is not implemented yet.")

def log_metrics(trainer, model, complement_loader, valset_loader, test_loader, compression_set_length, train_set_length, n_sigma, return_validation_loss=False):
    complement_res = trainer.validate(model=model, dataloaders=complement_loader)
    validation_res = trainer.validate(model=model, dataloaders=valset_loader)
    test_results = trainer.test(model, dataloaders=test_loader)

    if wandb.config['regression']:
        metrics = {'complement_loss' : complement_res[0]['validation_loss'],
                'validation_loss': validation_res[0]['validation_loss'],
                'test_loss': test_results[0]['test_loss'],
                'compression_set_size':compression_set_length}

        compute_real_valued_bounds(compression_set_length,
                                    n_sigma,
                                    train_set_length,
                                    complement_res[0]['validation_loss'],
                                    wandb.config['delta'],
                                    wandb.config['nbr_parameter_bounds'],
                                    metrics,
                                    min_val=wandb.config['min_val'],
                                    max_val=wandb.config['max_val'])
        wandb.log(metrics)
    else:
        metrics = {'complement_error' : complement_res[0]['validation_error'],
                'validation_error': validation_res[0]['validation_error'],
                'test_error': test_results[0]['test_error'],
                'compression_set_size':compression_set_length}

        compute_real_valued_bounds(compression_set_length,
                                    n_sigma,
                                    train_set_length,
                                    complement_res[0]['validation_error'],
                                    wandb.config['delta'],
                                    wandb.config['nbr_parameter_bounds'],
                                    metrics)
        k = int(complement_res[0]['validation_error']* (train_set_length-compression_set_length))
        compute_all_p2l_bounds(compression_set_length, train_set_length, k,  wandb.config['delta'], metrics)
        
        if wandb.config['clamping']:
            min_val_loss, max_val_loss = get_min_max_loss(wandb.config['min_probability'], wandb.config['n_classes'], wandb.config['clamp_method'])
            compute_real_valued_bounds(compression_set_length, n_sigma, train_set_length, complement_res[0]['validation_loss'], wandb.config['delta'],
                                        wandb.config['nbr_parameter_bounds'], metrics, min_val=min_val_loss, max_val=max_val_loss, prefix="CE")
        wandb.log(metrics)

    if return_validation_loss:
        return complement_res[0]['validation_loss']

def get_model_name_from_config(config, use_pactl=False, use_pbb=False):
    list_of_keys = ['dataset', 'seed', 'prior_size', 'pretraining_epochs','pretraining_lr', 'model_type',
                    'dropout_probability', 'optimizer', 'training_lr','weight_decay', 'min_probability',
                      'clamp_method', 'max_epochs']
    if use_pactl:
        list_of_keys += ['intrinsic_mode', 'intrinsic_dim', 'levels', 'rank']
    if use_pbb:
        list_of_keys += ['sigma_prior', 'mc_samples', 'kl_penalty']

    l = [str(config.get(name)) for name in list_of_keys]
    return_str = ""
    for elem in l :
        return_str += "_" + elem
    return_str += ".ckpt"
    return return_str[1:]
    

def get_best_model(dataset, model_type, seed, information_dict=None, number_of_seeds=5):
    best_val_error = 1
    best_config = []
    for _, _, filenames in os.walk("./baseline_logs"):
        counter = 0
        config_list = []
        sum_val_error = 0
        for file in sorted(filenames, key= lambda x: x[x.find("_", x.find("_") + 1)+1:]):
            if dataset in file and model_type in file:
                if counter != 0 and counter % number_of_seeds == 0:
                    if sum_val_error / number_of_seeds < best_val_error:
                        best_config = config_list.copy()
                        best_val_error = sum_val_error / number_of_seeds
                    config_list = []
                    sum_val_error = 0
                
                with open("./baseline_logs/" + file) as f:
                    d = json.load(f)
                    sum_val_error += d['validation_error']
                    config_list.append(d)
                counter += 1

    if len(best_config) == 0:
        if len(config_list) > 0:
            best_config = config_list
        else:
            raise RuntimeError("No configs were found for the baseline models.")
        
    config_best_model = sorted(best_config, key=lambda x:x['config']['seed'])[[1,2,3,4,42].index(seed)]
    if information_dict is not None:
        information_dict["best_model_complement_error"] = config_best_model["complement_error"]
        information_dict["best_model_validation_error"] = config_best_model["validation_error"]
        information_dict["best_model_test_error"] = config_best_model["test_error"]
        information_dict["best_model_complement_loss"] = config_best_model["complement_loss"]
        information_dict["best_model_validation_loss"] = config_best_model["validation_loss"]
        information_dict["best_model_test_loss"] = config_best_model["test_loss"]
    path_file = "./trained_models/" + get_model_name_from_config(config_best_model['config'])
    return load_pretrained_model(path_file, config_best_model['config'])

def compute_disagreement(compressed_model, disagreement_loader, config, information_dict, device, log_divergence=None):

    best_model = get_best_model(config['dataset'], config['model_type'], config['seed'], information_dict)
    pmin = compressed_model.pmin
    n_classes = compressed_model.model.n_classes
    if compressed_model.clamp_method == "clamp":
        pmin /= n_classes
        loss = ClampedCrossEntropyLoss(clamping=True, pmin=pmin, reduction='none')
    elif compressed_model.clamp_method == "smooth":
        loss = SmoothedCrossEntropyLoss(n_classes=n_classes, clamping=True, pmin=pmin, reduction='none')
    elif compressed_model.clamp_method == 'clip':
        loss = ClippedCrossEntropyLoss(pmin=pmin, reduction='none')
    elif compressed_model.clamp_method is None:
        loss = torch.nn.CrossEntropyLoss(reduction="none")
    else:
        raise ValueError(f"The clamping method {compressed_model.clamp_method} is not implemented yet.")
    
    sum_disagreement = 0
    sum_loss_disag = 0.0
    sum_softmax_disag = 0.0
    compressed_model.eval()
    compressed_model.to(device)
    best_model.eval()
    best_model.to(device)
    with torch.no_grad():
        for _,  batch in enumerate(tqdm(disagreement_loader, desc="Disagreement")):
            if isinstance(batch, transformers.tokenization_utils_base.BatchEncoding):
                x = {
                    'input_ids' : batch['input_ids'].to(device),
                    'labels': batch['labels'].to(device),
                    'attention_mask' : batch['attention_mask'].to(device)
                }
                y = batch['labels'].to(device)
                y_pred_comp = compressed_model.model(x)
                y_pred_best = best_model.model(x)
            else:
                x, y = batch
                x, y = x.to(device), y.to(device)
                y_pred_comp = compressed_model.model(x)
                y_pred_best = best_model.model(x)
                
            pred1 = torch.argmax(torch.nn.Softmax(dim=1)(y_pred_comp), dim=1)
            pred2 = torch.argmax(torch.nn.Softmax(dim=1)(y_pred_best), dim=1)
            disagreement = (pred1 != pred2).sum().item()
            sum_disagreement += disagreement

            loss1 = loss(y_pred_comp, y)
            loss2 = loss(y_pred_best, y)
            loss_disag = (loss1 - loss2).abs().sum().item()
            sum_loss_disag += loss_disag

            if compressed_model.clamp_method == "clamp":
                soft1 = torch.clamp(torch.nn.Softmax(dim=1)(y_pred_comp)[y], pmin)
                soft2 = torch.clamp(torch.nn.Softmax(dim=1)(y_pred_best)[y], pmin)
            elif compressed_model.clamp_method == "smooth":
                soft1 = (1-pmin) * torch.nn.Softmax(dim=1)(y_pred_comp)[y] + (pmin / n_classes)
                soft2 = (1-pmin) * torch.nn.Softmax(dim=1)(y_pred_best)[y] + (pmin / n_classes)
            elif compressed_model.clamp_method == "clip":
                soft1 = torch.zeros_like(y_pred_comp)
                norms = torch.linalg.norm(y_pred_comp.flatten(1), ord=1, dim=1)
                for i in range(y_pred_comp.shape[0]):
                    if norms[i] >= pmin:
                        soft1[i] = pmin * y_pred_comp[i].clone() / norms[i]
                    else:
                        soft1[i] = y_pred_comp[i]

                soft2 = torch.zeros_like(y_pred_best)
                norms = torch.linalg.norm(y_pred_best.flatten(1), ord=1, dim=1)
                for i in range(y_pred_best.shape[0]):
                    if norms[i] >= pmin:
                        soft2[i] = pmin * y_pred_best[i].clone() / norms[i]
                    else:
                        soft2[i] = y_pred_best[i]
            elif compressed_model.clamp_method is None:
                soft1 = torch.nn.Softmax(dim=1)(y_pred_comp)
                soft2 = torch.nn.Softmax(dim=1)(y_pred_best)

            disag = (soft1 - soft2).abs().sum().item()
            sum_softmax_disag += disag

    compute_disagreement_bounds(sum_disagreement, sum_loss_disag, sum_softmax_disag,
                                len(disagreement_loader.dataset), config, information_dict, log_divergence=log_divergence)


def compute_pac_bayes_disagreement(pb_model, nb_sampling, disagreement_loader, device,  config, information_dict):

    n_disagreement = len(disagreement_loader.dataset)

    best_model = get_best_model(config['dataset'], config['model_type'], config['seed'], information_dict)
    pmin = pb_model.pmin
    n_classes = pb_model.model.n_classes
    if pb_model.clamp_method == "clamp":
        pmin /= n_classes
        loss = ClampedCrossEntropyLoss(clamping=True, pmin=pmin, reduction='none')
    elif pb_model.clamp_method == "smooth":
        loss = SmoothedCrossEntropyLoss(n_classes=n_classes, clamping=True, pmin=pmin, reduction='none')
    elif pb_model.clamp_method == 'clip':
        loss = ClippedCrossEntropyLoss(pmin=pmin, reduction='none')
    else:
        raise ValueError(f"The clamping method {pb_model.clamp_method} is not implemented yet.")
    
    sum_disagreement = [ 0.0 for _ in range(nb_sampling)]
    sum_loss_disag = [ 0.0 for _ in range(nb_sampling)]
    sum_softmax_disag = [ 0.0 for _ in range(nb_sampling)]
    pb_model.eval()
    pb_model.to(device)
    best_model.eval()
    best_model.to(device)
    with torch.no_grad():
        with tqdm(total=len(disagreement_loader) * nb_sampling, desc='Disagreement') as pbar:
            for _,  (x, y) in enumerate(disagreement_loader):
                x, y = x.to(device), y.to(device)
                y_pred_best = best_model.model(x)
                pred_best = torch.argmax(torch.nn.Softmax(dim=1)(y_pred_best), dim=1)
                loss_best = loss(y_pred_best, y)

                if pb_model.clamp_method == "clamp":
                    soft_best = torch.clamp(torch.nn.Softmax(dim=1)(y_pred_best)[y], pmin)
                elif pb_model.clamp_method == "smooth":
                    soft_best = (1-pmin) * torch.nn.Softmax(dim=1)(y_pred_best)[y] + (pmin / n_classes)

                for i in range(nb_sampling):
                    y_pred_comp = pb_model.model(x)

                    pred1 = torch.argmax(torch.nn.Softmax(dim=1)(y_pred_comp), dim=1)
                    disagreement = (pred1 != pred_best).sum().item()
                    sum_disagreement[i] += disagreement / n_disagreement

                    loss1 = loss(y_pred_comp, y)
                    loss_disag = (loss1 - loss_best).abs().sum().item()
                    sum_loss_disag[i] += loss_disag / n_disagreement

                    if pb_model.clamp_method == "clamp":
                        soft1 = torch.clamp(torch.nn.Softmax(dim=1)(y_pred_comp)[y], pmin)
                    elif pb_model.clamp_method == "smooth":
                        soft1 = (1-pmin) * torch.nn.Softmax(dim=1)(y_pred_comp)[y] + (pmin / n_classes)

                    disag = (soft1 - soft_best).abs().sum().item()
                    sum_softmax_disag[i] += disag / n_disagreement
                    pbar.update(1)

    information_dict['disagreement'] = np.mean(sum_disagreement)
    information_dict['loss_disagreement'] = np.mean(sum_loss_disag)
    information_dict['softmax_disagreement'] = np.mean(sum_softmax_disag)

    return information_dict['disagreement'], information_dict['loss_disagreement'], information_dict['softmax_disagreement']


def compute_distillation_targets(dataset, model, device, collate_fn, config):
    dataloader = get_dataloader(dataset=dataset, batch_size=config['batch_size'], collate_fn=collate_fn)
    distillation_targets = []
    model.eval()
    model.to(device)
    with torch.no_grad():
        with tqdm(total=len(dataloader), desc='Distillation targets') as pbar:
            for _,  x in enumerate(dataloader):
                x = x.to(device)
                distillation_targets.append(torch.argmax(model.model(x), dim=1).cpu())
                pbar.update(1)
    model.cpu()
    return torch.cat(distillation_targets)


def create_distillation_targets(model, disagreement_set, device, collate_fn,  config):
        distillation_targets_path = "./distillation_targets/"
        if not os.path.isdir(distillation_targets_path):
            os.mkdir(distillation_targets_path)

        distillation_targets_file = distillation_targets_path +\
                        f"{config['dataset']}_{config['seed']}_{config['model_type']}_{config['training_lr']}_{config['optimizer']}.npy"
        if os.path.isfile(distillation_targets_file):
            distillation_targets = np.load(distillation_targets_file)
        else:
            distillation_targets = compute_distillation_targets(disagreement_set, model, device, collate_fn, config)
            np.save(distillation_targets_file, distillation_targets.numpy())

        disagreement_set.targets = distillation_targets