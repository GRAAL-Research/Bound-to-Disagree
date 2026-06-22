import torch
from transformers import DistilBertForSequenceClassification, DistilBertConfig, GPT2Config, GPT2ForSequenceClassification
from models.classification_model import ClassificationModel
import transformers
from copy import deepcopy
import loralib as lora
import os

class DistilBert(torch.nn.Module):
    def __init__(self, n_classes=2, dropout_probability=0.2):
        super().__init__()
        self.n_classes = n_classes
        self.dropout_probability = dropout_probability
        distilbert_config = DistilBertConfig(seq_classif_dropout=self.dropout_probability, num_labels=self.n_classes)
        distilbert_path = "./models/pretrained_models/distilbert_base_saved_on_computer"
        if os.path.isdir(distilbert_path):
            self.model = DistilBertForSequenceClassification.from_pretrained(pretrained_model_name_or_path=distilbert_path,
                                                                            config=distilbert_config)
        else:
            self.model = DistilBertForSequenceClassification.from_pretrained(pretrained_model_name_or_path="distilbert-base-uncased",
                                                                            config=distilbert_config)

    def forward(self, input):
        return self.model(**input).logits
    
class DistilBertLora(torch.nn.Module):
    def __init__(self, n_classes=10, rank=16, lora_alpha=1.0, init_model=None):
        super().__init__()
        self.n_classes = n_classes

        if init_model is not None:
            self.model = deepcopy(init_model.model)
        else: 
            self.model = DistilBert(n_classes=n_classes).model

        for name, module in self.model.named_modules():
                if isinstance(module, torch.nn.Linear):
                    temp = lora.Linear(in_features=module.in_features,
                                     out_features=module.out_features,
                                     r=rank,
                                     lora_alpha=lora_alpha,
                                     bias=(module.bias is not None),
                                     merge_weights=False)
                    if "." in name:
                        new_name = name[name.find(".")+1:]
                        mod = getattr(self.model, name[:name.find(".")])
                        init_mod = getattr(init_model.model, name[:name.find(".")])
                    else:
                        init_mod = init_model.model
                        mod = self.model
                        new_name = name
                    while "." in new_name:
                        mod = getattr(mod, new_name[:new_name.find(".")])
                        init_mod = getattr(init_mod, new_name[:new_name.find(".")])
                        new_name = new_name[new_name.find(".")+1:]
                    init_mod = getattr(init_mod, new_name)
                    temp.weight.data.copy_(init_mod.weight.data)
                    if module.bias is not None:
                        temp.bias.data.copy_(init_mod.bias.data)
                    setattr(mod, new_name, temp)

    def forward(self, input):
        return self.model(**input).logits

class GPT2(torch.nn.Module):
    def __init__(self, n_classes=2, dropout_probability=0.2):
        super().__init__()
        self.n_classes = n_classes
        self.dropout_probability = dropout_probability
        gpt_config = GPT2Config(seq_classif_dropout=self.dropout_probability, num_labels=self.n_classes)
        gpt2_path = "./models/pretrained_models/gpt2_base_saved_on_computer"
        if os.path.isdir(gpt2_path):
             self.model = GPT2ForSequenceClassification.from_pretrained(pretrained_model_name_or_path=gpt2_path,
                                                                          config=gpt_config)
        else:
            self.model = GPT2ForSequenceClassification.from_pretrained(pretrained_model_name_or_path="openai-community/gpt2",
                                                                            config=gpt_config)
        
        self.initialize_tokenizer()

    def forward(self, input):
        return self.model(**input).logits
    
    def initialize_tokenizer(self):
        gpt2_path = "./models/pretrained_models/tokenizer_gp2"
        if os.path.isdir(gpt2_path):
            tokenizer = transformers.GPT2Tokenizer.from_pretrained(gpt2_path)
        else:
            tokenizer = transformers.GPT2Tokenizer.from_pretrained("openai-community/gpt2")
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.model.resize_token_embeddings(len(tokenizer))
        self.model.config.pad_token_id = tokenizer.pad_token_id

    def replace_conv1d_for_linears(self):
        for name, module in self.model.named_modules():
            if isinstance(module, transformers.pytorch_utils.Conv1D):
                with torch.no_grad():
                    lin_layer = torch.nn.Linear(in_features=module.weight.shape[0],
                                                out_features=module.weight.shape[1],
                                                bias=module.bias is not None)
                    lin_layer.weight = torch.nn.Parameter(module.weight.T)
                    lin_layer.bias = module.bias
                new_name = name[name.find(".")+1:]
                mod = getattr(self.model, name[:name.find(".")])
                while "." in new_name:
                    mod = getattr(mod, new_name[:new_name.find(".")])
                    new_name = new_name[new_name.find(".")+1:]
                setattr(mod, new_name, lin_layer)

class GPT2Lora(torch.nn.Module):
    def __init__(self, n_classes=10, rank=16, lora_alpha=1.0, init_model=None):
        super().__init__()
        self.n_classes = n_classes

        if init_model is not None:
            self.model = deepcopy(init_model.model)
        else: 
            gpt = GPT2(n_classes=n_classes)
            gpt.replace_conv1d_for_linears()
            self.model = gpt.model

        for name, module in self.model.named_modules():
                if isinstance(module, torch.nn.Linear):
                    temp = lora.Linear(in_features=module.in_features,
                                     out_features=module.out_features,
                                     r=rank,
                                     lora_alpha=lora_alpha,
                                     bias=(module.bias is not None),
                                     merge_weights=False)
                    if "." in name:
                        new_name = name[name.find(".")+1:]
                        mod = getattr(self.model, name[:name.find(".")])
                        init_mod = getattr(init_model.model, name[:name.find(".")])
                    else:
                        init_mod = init_model.model
                        mod = self.model
                        new_name = name
                    while "." in new_name:
                        mod = getattr(mod, new_name[:new_name.find(".")])
                        init_mod = getattr(init_mod, new_name[:new_name.find(".")])
                        new_name = new_name[new_name.find(".")+1:]
                    init_mod = getattr(init_mod, new_name)
                    temp.weight.data.copy_(init_mod.weight.data)
                    if module.bias is not None:
                        temp.bias.data.copy_(init_mod.bias.data)
                    setattr(mod, new_name, temp)
                    
    def forward(self, input):
        return self.model(**input).logits

class ClassificationTransformerModel(ClassificationModel):

    def training_step(self, batch, batch_idx):
        y = batch['labels']
        return super().training_step((batch, y), batch_idx)
    
    def predict_step(self, batch, batch_idx):
        y = batch['labels']
        return super().predict_step((batch, y), batch_idx)
    
    def validation_step(self, batch, batch_idx):
        y = batch['labels']
        super().validation_step((batch, y), batch_idx)

    def test_step(self, batch, batch_idx):
        y = batch['labels']
        super().test_step((batch, y), batch_idx)