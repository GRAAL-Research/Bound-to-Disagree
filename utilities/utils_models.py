from models.linear_network import MnistMlp
from models.convolutional_network import MnistCnn, MnistLoraCnn, MnistProbCnn, Cifar10Cnn9l, Cifar10LoraCnn9l, Cifar10ProbCnn9l
from models.transformer import DistilBert, GPT2, ClassificationTransformerModel
from models.classification_model import ClassificationModel, ProbClassificationModel
from models.resnet import Cifar10Resnet, Cifar10LoraResnet, Cifar10ProbResnet
from models.transformer import DistilBertLora, GPT2Lora
import math
import torch
from pickle import UnpicklingError

def create_model(config):
    if config.get('prior_size', 0.0) == 0.0:
        lr = config.get('training_lr', None)
    else:
        lr = config.get('pretraining_lr', None)

    if config['dataset'] == "mnist":
        if config['model_type'] == "mlp":
            return ClassificationModel(MnistMlp(dataset_shape=784,
                                                n_classes=config['n_classes'],
                                                dropout_probability=config['dropout_probability']),
                                                optimizer=config['optimizer'],
                                                lr=lr,
                                                momentum=config['momentum'],
                                                batch_size=config['batch_size'],
                                                pmin=config['min_probability'],
                                                clamp_method=config["clamp_method"],
                                                weight_decay=config['weight_decay'],
                                                lr_scheduler=config['lr_scheduler']
                                                )
        elif config['model_type'] == "cnn":
            if config.get('rank', 0) == 0:
                model = MnistCnn(n_classes=config['n_classes'],
                                                dropout_probability=config['dropout_probability'])
            else:
                model = MnistLoraCnn(n_classes=config['n_classes'],
                                    dropout_probability=config['dropout_probability'],
                                    rank=config['rank'],
                                    lora_alpha=config['lora_alpha']
                                    )
            return ClassificationModel(model, optimizer=config['optimizer'],
                                                lr=lr,
                                                momentum=config['momentum'],
                                                batch_size=config['batch_size'],
                                                pmin=config['min_probability'],
                                                clamp_method=config["clamp_method"],
                                                weight_decay=config['weight_decay'],
                                                lr_scheduler=config['lr_scheduler']
                                                )
    elif config['dataset'] == "randomMnist":
        return ClassificationModel(MnistCnn(n_classes=config['n_classes'],
                                    dropout_probability=config['dropout_probability']),
                                    optimizer=config['optimizer'],
                                    lr=lr,
                                    momentum=config['momentum'],
                                    batch_size=config['batch_size'],
                                    pmin=config['min_probability'],
                                    clamp_method=config["clamp_method"],
                                    weight_decay=config['weight_decay'],
                                    lr_scheduler=config['lr_scheduler']
                                    )
    elif config['dataset'] == "cifar10":
        if config['model_type'] == "cnn":
            if config.get('rank', 0) == 0:
                model = Cifar10Cnn9l(n_classes=config['n_classes'],
                                    dropout_probability=config['dropout_probability'])
            else:
                model = Cifar10LoraCnn9l(n_classes=config['n_classes'],
                                        dropout_probability=config['dropout_probability'],
                                        rank=config['rank'],
                                        lora_alpha=config['lora_alpha']
                                    )
            return ClassificationModel(model,
                                        optimizer=config['optimizer'],
                                        lr=lr,
                                        momentum=config['momentum'],
                                        batch_size=config['batch_size'],
                                        pmin=config['min_probability'],
                                        clamp_method=config["clamp_method"],
                                        weight_decay=config['weight_decay'],
                                        lr_scheduler=config['lr_scheduler']
                                        )
        elif config['model_type'] == "resnet":
            if config.get('rank', 0) == 0:
                model = Cifar10Resnet(n_classes=config['n_classes'])
            else:
                model = Cifar10LoraResnet(n_classes=config['n_classes'],
                                          rank=config['rank'],
                                            lora_alpha=config['lora_alpha'])
            return ClassificationModel(model,
                                        optimizer=config['optimizer'],
                                        lr=lr,
                                        momentum=config['momentum'],
                                        batch_size=config['batch_size'],
                                        pmin=config['min_probability'],
                                        clamp_method=config["clamp_method"],
                                        weight_decay=config['weight_decay'],
                                        lr_scheduler=config['lr_scheduler'])
    elif config['dataset'] == "amazon":
        if config['model_type'] == "MobileBERT":
            return ClassificationTransformerModel(MobileBERT(n_classes=config['n_classes'],
                                                dropout_probability=config['dropout_probability']),
                                    optimizer=config['optimizer'],
                                    lr=lr,
                                    momentum=config['momentum'],
                                    batch_size=config['batch_size'],
                                    huber_delta=config.get('huber_delta',0.2))
        elif config['model_type'] == "DistilBERT":
            return ClassificationTransformerModel(DistilBert(n_classes=config['n_classes'],
                                                dropout_probability=config['dropout_probability']),
                                                optimizer=config['optimizer'],
                                                lr=lr,
                                                momentum=config['momentum'],
                                                batch_size=config['batch_size'],
                                                huber_delta=config.get('huber_delta',0.2))
        elif config['model_type'] == "GPT2":
            return ClassificationTransformerModel(GPT2(n_classes=config['n_classes'],
                                                dropout_probability=config['dropout_probability']),
                                                optimizer=config['optimizer'],
                                                lr=lr,
                                                momentum=config['momentum'],
                                                batch_size=config['batch_size'],
                                                huber_delta=config.get('huber_delta',0.2))
    elif config['dataset'] in ["concrete", "airfoil", "parkinson", "infrared", "powerplant"]:
        if config['model_type'] == "tree":
            return RegressionTreeModel(RegressionTree(
                max_depth=config['max_depth'],
                min_samples_split=config['min_samples_split'],
                min_samples_leaf=config['min_samples_leaf'],
                seed=config['seed'],
                ccp_alpha=config['ccp_alpha']
            ))
        elif config['model_type'] == "forest":
            return RegressionTreeModel(RegressionForest(
                n_estimators=config['n_estimators'],
                max_depth=config['max_depth'],
                min_samples_split=config['min_samples_split'],
                min_samples_leaf=config['min_samples_leaf'],
                seed=config['seed'],
                ccp_alpha=config['ccp_alpha']
            ))
    elif config['dataset'] == "moons":
        if config['model_type'] == "tree":
            return ClassificationTreeModel(ClassificationTree(
                n_classes=config['n_classes'],
                max_depth=config['max_depth'],
                min_samples_split=config['min_samples_split'],
                min_samples_leaf=config['min_samples_leaf'],
                seed=config['seed'],
                ccp_alpha=config['ccp_alpha']
            ))
        elif config['model_type'] == "forest":
            return ClassificationTreeModel(ClassificationForest(
                n_classes=config['n_classes'],
                n_estimators=config['n_estimators'],
                max_depth=config['max_depth'],
                min_samples_split=config['min_samples_split'],
                min_samples_leaf=config['min_samples_leaf'],
                seed=config['seed'],
                ccp_alpha=config['ccp_alpha']
            ))
    
    raise NotImplementedError(f"Model type = {config['model_type']} with dataset {config['dataset']} is not implemented yet.")


def create_probabilistic_net(config, model, device):
    rho_prior = math.log(math.exp(config.get('sigma_prior', 0.01))-1.0)

    if config['dataset'] == "mnist":
        if config['model_type'] == "cnn":
            return MnistProbCnn(config['n_classes'], rho_prior=rho_prior, device=device, init_net=model)
    elif config['dataset'] == "cifar10":
        if config['model_type'] == "cnn":
            return Cifar10ProbCnn9l(config['n_classes'], rho_prior=rho_prior, device=device, init_net=model)
        elif config['model_type'] == "resnet":
            return Cifar10ProbResnet(config['n_classes'], rho_prior=rho_prior, device=device, init_net=model)
    
    raise NotImplementedError(f"Model type = {config['model_type']} with dataset {config['dataset']} is not implemented yet.")

def create_probabilistic_model(config, model, pbbobj, min_max_vals):
    net = create_probabilistic_net(config, model.model, pbbobj.device)
    return ProbClassificationModel(model=net, pbbobj=pbbobj, min_val=min_max_vals[0], max_val=min_max_vals[1], optimizer=model.optimizer,
                                    lr=model.lr, momentum=model.momentum, batch_size=model.batch_size, pmin=model.pmin,
                                clamp_method=model.clamp_method, weight_decay=model.weight_decay,
                                  lr_scheduler=model.lr_scheduler, nb_batches=model.nb_batches)
    
def create_lora_model(config, model):
    for param in model.parameters():
        param.requires_grad = False
        
    if config['dataset'] == "mnist":
        return MnistLoraCnn(n_classes=config['n_classes'],
                             rank=config['rank'], lora_alpha=config['lora_alpha'], init_model=model)
    elif config['dataset'] == "cifar10":
        if config['model_type'] == "cnn":
            return Cifar10LoraCnn9l(n_classes=config['n_classes'],
                             rank=config['rank'], lora_alpha=config['lora_alpha'], init_model=model)
        elif config['model_type'] == "resnet":
            return Cifar10LoraResnet(n_classes=config['n_classes'],
                             rank=config['rank'], lora_alpha=config['lora_alpha'], init_model=model)
    elif config['dataset'] == "amazon":
        if config['model_type'] == "DistilBERT":
            return DistilBertLora(n_classes=config['n_classes'],
                             rank=config['rank'], lora_alpha=config['lora_alpha'], init_model=model)
        elif config['model_type'] == "GPT2":
            return GPT2Lora(n_classes=config['n_classes'],
                             rank=config['rank'], lora_alpha=config['lora_alpha'], init_model=model)
    raise NotImplementedError(f"The model {config['model_type']} is not implemented for the dataset {config['dataset']}")

def load_pretrained_model(checkpoint_path, config):
    try:
        if config.get('regression', False):
            if config['model_type'] in ['tree', 'forest']:
                return RegressionTreeModel.load_from_checkpoint(checkpoint_path)
        else:
            if config['model_type'] in ['mlp', 'cnn', 'resnet']:
                return ClassificationModel.load_from_checkpoint(checkpoint_path)
            elif config['model_type'] in ["MobileBERT", "DistilBERT", "GPT2"]:
                return ClassificationTransformerModel.load_from_checkpoint(checkpoint_path)
    except UnpicklingError:
        return _load_pretrained_model(checkpoint_path, config)
    setting = "regression" if config['regression'] else "classification"
    raise NotImplementedError(f"Loading checkpoints for a {config['model_type']} in a {setting} setting is not supported yet.")

def _load_pretrained_model(checkpoint_path, config):
    model = create_model(config)
    ckpt = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    if hasattr(model.model, 'replace_conv1d_for_linears'):
        model.model.replace_conv1d_for_linears()
    return model