# New generalization bounds for deep learning via disagreement

This is the repository associated to the UAI 2026 paper : 

**Bound to Disagree: Generalization Bounds via Certifiable Surrogates**

## Citation information
If you use our code, please cite our paper : 

```
Bazinet, M., Zantedeschi, V., & Germain, P. (2026). Bound to Disagree : Generalization Bounds via Certifiable Surrogates. In The Forty-Second Annual Conference on Uncertainty in Artificial Intelligence.
```

```
@inproceedings{bazinet2026bound,
title={Bound to Disagree : Generalization Bounds via Certifiable Surrogates},
author={Mathieu Bazinet and Valentina Zantedeschi and Pascal Germain},
booktitle={Forty-Second Annual Conference on Uncertainty in Artificial Intelligence},
year={2026},
url={https://openreview.net/forum?id=CZeG5HW1Y5}
}
```


## Running the code

To clone the repository, please use the following command :  
`git clone --recurse-submodules https://github.com/GRAAL-Research/Bound-to-Disagree.git`

We build our code base on the DeepCore library [1], the PACTL library [2] and PBB library [3], which we add as submodules.

To run the experiments, please use the following commands : 

- P2L : `python main_p2l.py -d <dataset-name>`
- Coresets : `python main_coreset.py -d <dataset-name>`
- Model compression : `python main_pactl.py -d <dataset-name>`
- PAC-Bayes : `python main_pbb.py -d <dataset-name>`
- Quantization : `python main_quantization.py -d <dataset-name>`

The code for the plots can be found in `result_plots`. 

## Python installations
The experiments were run on three different devices. The target neural networks, the experiments with Pick-To-Learn and the quantization experiments were computed on a computer with Python 3.12.3 and a NVIDIA GeForce RTX 4090. The coreset experiments and the model compression experiments were computed with Python 3.12.4 and a NVidia H100 SXM5. Finally, the PAC-Bayesian experiments and the model distillation experiments were computed with Python 3.12.4 and a NVidia A100 SXM4.

Use `requirements/requirements_amazon.txt` to run the experiments on Amazon polarity. To use the PACTL library, use `requirements/requirements_pactl.txt`. Otherwise, use `requirements/requirements.txt`. 

## Special setup for DeepCore library

The DeepCore library necessits a special setup, so please follow the steps provided. The first two commands are necessary to use our specific neural network architectures. The last two commands are necessary to fix bugs on my computers, and possibly all computers.

- Add file `custom_deepcore_networks.py` to `DeepCore/deepcore/nets`
- Add `from .custom_deepcore_networks import *` in `DeepCore/deepcore/nets/__init__.py`.
- Change line 92 to `embdeddings.append(model.embedding_recorder.embedding.flatten(1).cpu().detach().numpy())` in `DeepCore/deepcore/methods/cal.py`.
- Change line 36 to `self.forgetting_events[torch.tensor(batch_inds).to(self.args.device)[(self.last_acc[batch_inds]-cur_acc)>0.01]]+=1.` in `DeepCore/deepcore/methods/forgetting.py`.

## Special setup for PACTL library

When running experiments with Amazon polarity, we need to modify the inference process in the file `Pactl/pactl/bounds/quantize_fns.py`. At the top of the file, add `import transformers`. 

Moreover, you need to change the lines 138-139 and 166-167 for this :
```
for i, batch in tqdm(enumerate(train_loader), leave=False):
    if isinstance(batch, transformers.tokenization_utils_base.BatchEncoding):
        X = {
            'input_ids' : batch['input_ids'].to(device),
            'labels': batch['labels'].to(device),
            'attention_mask' : batch['attention_mask'].to(device)
        }
        Y = batch['labels'].to(device)
    else:
        X, Y = batch
        X, Y = X.to(device), X.to(device)
```


## References
1. Guo, C., Zhao, B., & Bai, Y. (2022, July). Deepcore: A comprehensive library for coreset selection in deep learning. In International Conference on Database and Expert Systems Applications (pp. 181-195). Cham: Springer International Publishing.
2. Lotfi, S., Finzi, M., Kapoor, S., Potapczynski, A., Goldblum, M., & Wilson, A. G. (2022). PAC-Bayes compression bounds so tight that they can explain generalization. Advances in Neural Information Processing Systems, 35, 31459-31473.
3. Pérez-Ortiz, M., Rivasplata, O., Shawe-Taylor, J., & Szepesvári, C. (2021). Tighter risk certificates for neural networks. Journal of Machine Learning Research, 22(227), 1-40.