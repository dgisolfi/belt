"""Translation Datasets

Use hugging face datasets and tokenizers to create 
language word pairings for translation task

Author(s)
---------
Daniel Nicolas Gisolfi <dgisolfi3@gatech.edu>
"""

import os
from dataclasses import dataclass
from datasets import load_dataset

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

DATASETS = {
    "opus_books": "Helsinki-NLP/opus_books",
    "multi30k": "bentrevett/multi30k"
}


@dataclass(frozen=True)
class TranslationData:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    source_tokenizer: PreTrainedTokenizerBase
    target_tokenizer: PreTrainedTokenizerBase
    source_lang: str
    target_lang: str


def lang_pairs(row, source_lang, target_lang):
    translation = row.get("translation")
    if not isinstance(translation, dict):
        translation = row
        
    source = translation.get(source_lang) 
    target = translation.get(target_lang)
    
    return str(source), str(target)


def load_translation_data(
    dataset_name="multi30k",
    dataset_config=None,
    source_lang="en",
    target_lang="de",
    split="train",
    sample_size=2000,
    val_size=200,
    test_size=200,
    batch_size=32,
    max_length=32,
):
    dataset_name = DATASETS[dataset_name]
    raw = (
        load_dataset(dataset_name, dataset_config)
        if dataset_config
        else load_dataset(dataset_name)
    )

    pairs = []
    for row in raw:
        pair = lang_pairs(row, source_lang, target_lang)
        if pair:
            pairs.append(pair)


    train_pairs = pairs[: -(val_size + test_size)]
    val_pairs = pairs[-(val_size + test_size) : -test_size]
    test_pairs = pairs[-test_size:]




