from datasets import load_dataset, load_from_disk
from transformers import DistilBertTokenizerFast, DataCollatorWithPadding, MobileBertTokenizerFast, GPT2Tokenizer
import os
from utilities.utils_datasets import CustomDataset
from functools import partial


def get_tokenizer(model):
    if model == "DistilBERT":
        distilbert_path = "./models/pretrained_models/tokenizer_distilbert"
        if os.path.isdir(distilbert_path):
            return DistilBertTokenizerFast.from_pretrained(distilbert_path)
        else:
            return DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    elif model == "GPT2":
        gpt2_path = "./models/pretrained_models/tokenizer_gpt2"
        if os.path.isdir(gpt2_path):
            tokenizer = GPT2Tokenizer.from_pretrained(gpt2_path)
        else:
            tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        return tokenizer
    else:
        raise NotImplementedError(f"The model {model} is not implemented for the tokenizer in Amazon Polarity")

def collate_function(input, data_collator=None):
    return data_collator([{'input_ids':i[0], 'labels':i[1]} for i in input])
    
def load_amazon_polarity(n_shards=1, model="DistilBERT"):
    file_path = "./amazon_polarity"
    if not os.path.isdir(file_path):
        os.mkdir(file_path)

    tokenizer = get_tokenizer(model)

    file_name = f"/amazon_polarity_dataset_{model}"
    file_path = file_path + file_name
    if not os.path.isdir(file_path):
        ds = load_dataset("mteb/amazon_polarity")
        tokenized_datasets = ds.map(lambda dataset: tokenizer(dataset['text'], truncation=True), batched=True)
        tokenized_datasets.set_format(type="torch", columns=["input_ids", "label"])
        tokenized_datasets.save_to_disk(file_path)
    else:
        tokenized_datasets = load_from_disk(file_path)
    
    train_tokenized = tokenized_datasets['train']
    test_tokenized = tokenized_datasets['test']
    if n_shards != 1:
        train_tokenized = train_tokenized.shard(num_shards=n_shards, index=0)
    
    data_collator = DataCollatorWithPadding(tokenizer)
    collate_fn = partial(collate_function, data_collator=data_collator)
    
    train_set = CustomDataset(data=train_tokenized['input_ids'],
                            targets=train_tokenized['label'],
                            transform=None,
                            real_targets=False,
                            is_an_image=False)
    test_set = CustomDataset(data=test_tokenized['input_ids'],
                        targets=test_tokenized['label'],
                        transform=None,
                        real_targets=False,
                        is_an_image=False)
    
    return train_set, test_set, collate_fn


