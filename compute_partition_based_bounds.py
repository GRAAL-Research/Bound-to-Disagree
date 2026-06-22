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
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin

def compute_clusters(n_clusters, train_set, config, disag_set=None, random_seed=None):
    file_path = "./clustering/"
    if not os.path.isdir(file_path):
        os.mkdir(file_path)
    file_name = file_path + f"clustering_{n_clusters}_{config['dataset']}_{cluster_method}_{config['seed']}_{random_seed}.npy"

    if os.path.isfile(file_name):
        clustering = np.load(file_name)
    else:
        if config['cluster_method'] == "kmeans":
            clf = KMeans(n_clusters=n_clusters, init='random', n_init=1, random_state=config['seed'])
            clustering = clf.fit_predict(train_set)
        elif config['cluster_method'] == "random":
            rng = np.random.default_rng(random_seed)
            centroids = rng.uniform(0, 1, size=(n_clusters, ) + train_set[0].shape)
            clustering = pairwise_distances_argmin(train_set, centroids)
        elif config['cluster_method'] == 'disagreement':
            clf = KMeans(n_clusters=n_clusters, init='random', n_init=1, random_state=config['seed'])
            clf.fit(disag_set)
            clustering = clf.predict(train_set)
        np.save(file_name, clustering)

    clusters, numbers = np.unique(clustering, return_counts=True)

    T = clusters.shape[0]
    return T, numbers

def compute_bound(n_clusters, T, numbers, gamma, n, delta, train_error, alpha, C):

    u_hat = gamma / (2*n) + gamma**2 * np.sqrt((2/n) * np.log(2*n_clusters/delta))
    u_hat += gamma**2 / 2 * torch.tensor(numbers).div(n).pow(2).sum().item()

    g = C * (np.sqrt(2)+1) * np.sqrt((T/n) * np.log(2*2*n_clusters/delta)) + 2*C*T*np.log(2*2*n_clusters/delta)/n

    return train_error + C * np.sqrt(u_hat * alpha * np.log(gamma)) + g

def compute_partition_based_bounds(config):

    file_path = "./partition_based_bounds/"
    if not os.path.isdir(file_path):
        os.mkdir(file_path)

    file_name = f"{config['dataset']}_{config['model_type']}_{config['seed']}_{config['min_probability']}_{config['clamp_method']}_{config['cluster_method']}.json"
    file_name = file_path + file_name
    if os.path.isfile(file_name):
        with open(file_name) as json_file:
            d = json.load(json_file)
            print(d['zero_one_bound'], d['cross_entropy_bound'])
            return 

    information_dict = {}
    best_model = get_best_model(config['dataset'], config['model_type'], config['seed'], information_dict)

    train_error = information_dict['best_model_complement_error']
    
    seed_everything(config['seed'], workers=True)
    train_set, _, collate_fn = load_dataset(config)

    train_set, _ = split_train_validation_dataset(train_set, config['validation_size'])
    train_set, disag_set = split_train_validation_dataset(train_set, config.get('disagreement_size',0.2))
    trainset_loader = get_dataloader(dataset=train_set, batch_size=config['batch_size'], collate_fn=collate_fn)

    n = len(train_set)

    pmin = config['min_probability']
    if config['clamp_method'] == "clamp":
        pmin /= config['n_classes']
        loss = ClampedCrossEntropyLoss(clamping=True, pmin=pmin, reduction='sum')
    elif config['clamp_method'] == "smooth":
        loss = SmoothedCrossEntropyLoss(n_classes=config['n_classes'], clamping=True, pmin=pmin, reduction='sum')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    best_model.eval()
    best_model.to(device)
    train_loss = 0.0
    with torch.no_grad():
        for _,  (x, y) in enumerate(tqdm(trainset_loader, desc="Training loss")):
            x, y = x.to(device), y.to(device)
            y_hat = best_model.model(x)
            train_loss += loss(y_hat, y).detach().item()/n
    best_model.cpu()

    train_set = train_set.data.flatten(1).numpy() / 255
    disag_set = disag_set.data.flatten(1).numpy() / 255
    
    cluster_size_list = [5, 10, 20, 50, 100, 200]
    random_seeds_list = [40, 41, 42, 43, 44] if config['cluster_method'] == 'random' else [None]
    zero_one_bound_list = []
    with tqdm(total=len(cluster_size_list) * 4 * len(random_seeds_list), desc='Zero-one bound') as pbar:
        for n_clusters in cluster_size_list:
            for random_seed in random_seeds_list:
                T, numbers = compute_clusters(n_clusters, train_set, config, disag_set=disag_set, random_seed=random_seed)

                alpha_list = [20, 50, 100, int(n*(n_clusters+n)/ (n_clusters *(4*n-3)))]
                for  alpha in alpha_list:
                    delta = config['delta'] / (len(cluster_size_list) * len(alpha_list)* len(random_seeds_list))
                    gamma = delta**(-1/alpha)
                    assert gamma > 1.0

                    bound = compute_bound(n_clusters, T, numbers, gamma, n, delta, train_error, alpha, 1)
                    zero_one_bound_list.append(bound)

                    pbar.update(1)

    cross_entropy_list = []
    with tqdm(total=len(cluster_size_list) * len(alpha_list)* len(random_seeds_list), desc='Cross-entropy bound') as pbar:
        for n_clusters in cluster_size_list:
            for random_seed in random_seeds_list:
                T, numbers = compute_clusters(n_clusters, train_set, config, disag_set=disag_set, random_seed=random_seed)

                alpha_list = [20, 50, 100, int(n*(n_clusters+n)/ (n_clusters *(4*n-3)))]
                for alpha in alpha_list:
                    delta = config['delta'] / (len(cluster_size_list) * len(alpha_list) * len(random_seeds_list))
                    gamma = delta**(-1/alpha)
                    assert gamma > 1.0

                    min_val, max_val = get_min_max_loss(config['min_probability'], config['n_classes'], config['clamp_method'])

                    bound = compute_bound(n_clusters, T, numbers, gamma, n, delta, train_loss, alpha, max_val - min_val)
                    cross_entropy_list.append(bound)
                    pbar.update(1)

    information_dict['complement_loss'] = train_loss
    information_dict['config'] = config
    information_dict['zero_one_bound'] = min(zero_one_bound_list)
    information_dict['cross_entropy_bound'] = min(cross_entropy_list)

    with open(file_name, "w") as outfile: 
        json.dump(information_dict, outfile)
    print(information_dict['zero_one_bound'], information_dict['cross_entropy_bound'])
    
if __name__ == "__main__":
    for cluster_method in ['random', 'disagreement', 'kmeans']:
            for dataset in ['mnist', 'cifar10']:
                for model_type in ['cnn', 'resnet']:
                    for seed in [1,2,3,4,42]:
                        for clamp_method in ['smooth', 'clamp']:
                            for min_probability in [1e-3, 1e-4, 1e-5]:
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
                                    'delta':0.005,
                                    'clamp_method':clamp_method,
                                    'min_probability':min_probability,
                                    'cluster_method': cluster_method
                                }
                                compute_partition_based_bounds(config)
 
