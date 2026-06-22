from bounds.kl_inv import kl_inv
import torch
from models.classification_model import ClampedCrossEntropyLoss, SmoothedCrossEntropyLoss, ClippedCrossEntropyLoss
from tqdm import tqdm, trange
import math

def monte_carlo_sampling(model, nb_sampling, test_loader, device, config):
    error = 0.0
    cross_entropy = 0.0

    pmin = config['min_probability']
    n_classes = model.n_classes
    if config['clamp_method'] == "clamp":
        pmin /= n_classes
        loss = ClampedCrossEntropyLoss(clamping=True, pmin=pmin, reduction='sum')
    elif config['clamp_method'] == "smooth":
        loss = SmoothedCrossEntropyLoss(n_classes=n_classes, clamping=True, pmin=pmin, reduction='sum')
    elif config['clamp_method'] == 'clip':
        loss = ClippedCrossEntropyLoss(pmin=pmin, reduction='sum')
    else:
        raise ValueError(f"The clamping method {config['clamp_method']} is not implemented yet.")

    model.set_sampling_mode(True)
    model.eval()
    model.to(device)
    with torch.no_grad():
        with tqdm(total=len(test_loader) * nb_sampling, desc='MC Sampling') as pbar:
            for _,  (x, y) in enumerate(test_loader):
                x, y = x.to(device), y.to(device)
                error_mc = 0.0
                ce_mc = 0.0
                for _ in range(nb_sampling):
                    y_hat = model(x)
                    error_mc += (torch.argmax(y_hat, dim=1) != y).sum()
                    ce_mc += loss(y_hat, y)
                    pbar.update(1)

                error += error_mc/nb_sampling
                cross_entropy += ce_mc / nb_sampling
                
    
    model.train()
    model.cpu()

    return error/len(test_loader.dataset), cross_entropy/len(test_loader.dataset)


def compute_pac_bayes_bound(model, loss, min_val, max_val, delta, train_set_size, nb_sampling):
    new_delta = delta/2
    normalized_loss = (loss - min_val) / (max_val - min_val)

    epsilon_1 = math.log(1/new_delta)/nb_sampling
    first_bound = kl_inv(normalized_loss, epsilon_1, "MAX")

    kl = model.model.compute_kl().item()
    epsilon_2 = (kl + math.log(2 * math.sqrt(train_set_size)/new_delta)) / train_set_size

    second_bound = kl_inv(first_bound, epsilon_2, "MAX")

    return min_val + (max_val - min_val) * second_bound, kl


def compute_pac_bayes_disagreement_bound(sum_disagreement, sum_loss_disag, sum_softmax_disag, min_max_vals, n_disagreement, delta, pmin, n_classes, information_dict):
    new_delta = delta / 2
    min_val = min_max_vals[0]
    max_val = min_max_vals[1]
    nb_samples = len(sum_disagreement)

    epsilon = math.log(1/new_delta) / n_disagreement
    first_bound = kl_inv(sum_disagreement, epsilon, "MAX")
    disag_bound = first_bound/nb_samples

    second_bound = kl_inv(sum_loss_disag / (max_val - min_val), epsilon, "MAX")
    disag_loss_bound = second_bound/nb_samples

    if (n_classes/pmin) * sum_softmax_disag > 1:
        third_bound = 1
    else:
        third_bound = kl_inv(sum_softmax_disag/ (max_val - min_val), epsilon, "MAX")
    disag_softmax_bound = third_bound/nb_samples

    epsilon = math.log(1/new_delta) / nb_samples
    information_dict['disagreement_bound'] = kl_inv(disag_bound, epsilon, "MAX")
    information_dict['disagreement_loss_kl'] = (max_val - min_val) * kl_inv(disag_loss_bound, epsilon, "MAX")
    information_dict['disagreement_softmax_kl'] = (max_val - min_val) * kl_inv(disag_softmax_bound, epsilon, "MAX")

    information_dict['full_disagreement'] = information_dict['kl_bound'] + information_dict['disagreement_bound']
    information_dict['full_disagreement_loss'] = information_dict['CE_kl_bound']  + information_dict['disagreement_loss_kl']
    information_dict['full_disagreement_softmax'] = information_dict['CE_kl_bound']  +information_dict['disagreement_softmax_kl']


