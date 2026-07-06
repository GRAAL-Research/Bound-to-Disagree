from bounds.classical_bounds import brute_force_binomial_tail_inversion, binomial_approximation
import numpy as np
from bounds.kl_inv import kl_inv

def compute_disagreement_bounds(disagreement, loss_disagreement, softmax_disagreement,
                            disagreement_set_size, config, information_dict, log_divergence=None):
    information_dict['disagreement'] = disagreement/disagreement_set_size
    information_dict['loss_disagreement'] = loss_disagreement/disagreement_set_size
    information_dict['softmax_disagreement'] = softmax_disagreement/disagreement_set_size

    # add p2l bounds
    log_delta = np.log(config['delta'])
    if log_divergence is not None:
        log_delta -= log_divergence

    information_dict['disagreement_brute_force'] = brute_force_binomial_tail_inversion(disagreement,
                                                                                    disagreement_set_size, 
                                                                                    log_delta)
    information_dict['disagreement_approximation'] = binomial_approximation(disagreement,
                                                                            disagreement_set_size,
                                                                            log_delta)
    information_dict['disagreement_kl'] = kl_inv(disagreement/disagreement_set_size, -log_delta/disagreement_set_size , "MAX")

    information_dict['full_disagreement_bound_brute_force'] = information_dict['binomial_approximation_shah']\
        +  information_dict['disagreement_brute_force']
    information_dict['full_disagreement_bound_approx'] = information_dict['binomial_approximation_shah']\
          + information_dict['disagreement_approximation'] 

    if information_dict.get('p2l_bound', None) is not None:
        information_dict['full_disagreement_bound_p2l'] = information_dict['p2l_bound']\
            +  information_dict['disagreement_brute_force']
        information_dict['full_disagreement_bound_p2l_approx'] = information_dict['p2l_bound']\
            + information_dict['disagreement_approximation']

    if config.get('selection', None) is not None and config['selection'] == "Uniform":
        information_dict['full_disagreement_test_set'] = information_dict['test_set_bound_brute']\
              + information_dict['disagreement_brute_force']
        information_dict['full_disagreement_binomial_approx'] = information_dict['test_set_bound_binomial_approx']\
              + information_dict['disagreement_approximation'] 

    epsilon = -log_delta/disagreement_set_size
    if config['clamp_method'] == "clamp":
        n_classes = config['n_classes']
        pmin = config['min_probability']        
        
        beta = n_classes / pmin

        loss = loss_disagreement/(np.log(beta) * disagreement_set_size)
        if 1.0 < loss <= 1+1e-10:
            loss = 1.0
        elif -1e-5 < loss < 0.0:
            loss = 0.0
        print("Loss: ", loss)
        information_dict['disagreement_loss_kl'] = np.log(beta) * kl_inv(loss, epsilon , "MAX") 

        loss = beta * softmax_disagreement/( np.log(beta) * disagreement_set_size)
        if loss > 1:
            information_dict['disagreement_softmax_kl'] = np.log(beta)
        else:
            information_dict['disagreement_softmax_kl'] = np.log(beta) * kl_inv(loss, epsilon , "MAX")
        
    elif config['clamp_method'] == "smooth":
        n_classes = config['n_classes']
        pmin = config['min_probability']
        beta = n_classes / pmin

        log_beta = np.log(1+(1-pmin)*beta)
        loss = loss_disagreement/(log_beta * disagreement_set_size)
        if 1.0 < loss <= 1+1e-10:
            loss = 1.0
        elif -1e-5 < loss < 0.0:
            loss = 0.0
        print("Loss: ", loss)
        information_dict['disagreement_loss_kl'] = log_beta * kl_inv(loss, epsilon , "MAX") 

        loss = beta *  softmax_disagreement/( log_beta * disagreement_set_size)
        print("Loss: ", loss)
        if loss > 1:
            information_dict['disagreement_softmax_kl'] = log_beta
        else:
            information_dict['disagreement_softmax_kl'] = log_beta * kl_inv(loss, epsilon , "MAX") 
    elif config['clamp_method'] == "clip":
        n_classes = config['n_classes']
        pmin = config['min_probability']
        max_val_loss = np.log(1+(n_classes-1)*np.exp(pmin)) 
        min_val_loss = np.log(1+(n_classes-1)*np.exp(-pmin)) 

        loss = (loss_disagreement - min_val_loss)/ (max_val_loss - min_val_loss) / disagreement_set_size
        information_dict['disagreement_loss_kl'] = (max_val_loss - min_val_loss) * kl_inv(loss, epsilon , "MAX") + min_val_loss
        
        loss = softmax_disagreement / (2 * pmin * disagreement_set_size)
        information_dict['disagreement_softmax_kl'] = 2 * pmin * kl_inv(loss, epsilon, "MAX")

    elif config['clamp_method'] is None:
        loss = softmax_disagreement/( 2 * disagreement_set_size)
        information_dict['disagreement_softmax_huber'] = 2 * config['huber_delta'] * kl_inv(loss, epsilon , "MAX") 

    if information_dict.get("test_set_CE_bound", None) is not None:
        information_dict['full_disagreement_loss'] = information_dict["test_set_CE_bound"] + information_dict['disagreement_loss_kl']
        information_dict['full_disagreement_softmax'] = information_dict["test_set_CE_bound"] + information_dict['disagreement_softmax_kl']
    elif information_dict.get("huber_kl_bound", None) is not None:
        information_dict['full_disagreement_softmax'] = information_dict["huber_kl_bound"] + information_dict['disagreement_softmax_huber']
    else:
        information_dict['full_disagreement_loss'] = information_dict["CE_kl_bound"] + information_dict['disagreement_loss_kl']
        information_dict['full_disagreement_softmax'] = information_dict["CE_kl_bound"] + information_dict['disagreement_softmax_kl']