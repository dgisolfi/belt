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
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

SUPPORTED_DATASETS = [
    "Helsinki-NLP/opus_books",
    "bentrevett/multi30k",
]


@dataclass(frozen=True)
class TranslationData:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    source_tokenizer: PreTrainedTokenizerBase
    target_tokenizer: PreTrainedTokenizerBase
    source_lang: str
    target_lang: str


class TranslationDataset(Dataset):
    def __init__(self, pairs, source_tokenizer, target_tokenizer, max_length):
        self.pairs = pairs
        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        source, target = self.pairs[index]
        src_ids = self.source_tokenizer(
            source,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids.squeeze(0)
        tgt_ids = self.target_tokenizer(
            target,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids.squeeze(0)
        return src_ids, tgt_ids


def _translation_loader(
    pairs, source_tokenizer, target_tokenizer, max_length, batch_size, shuffle
):
    dataset = TranslationDataset(pairs, source_tokenizer, target_tokenizer, max_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


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
    seed=42,
    shuffle=True,
    suppress_symlink_warning=True,
    no_cache=False,
):
    if suppress_symlink_warning:
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    if no_cache:
        os.environ.pop("HF_DATASETS_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)

    model_name = f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"
    source_tokenizer = AutoTokenizer.from_pretrained(
        model_name, force_download=no_cache
    )
    # opus-mt is directional, one tokenizer covers both sides
    target_tokenizer = source_tokenizer

    if dataset_name not in SUPPORTED_DATASETS:
        raise ValueError(
            f"Unsupported dataset '{dataset_name}'. Choose from: {SUPPORTED_DATASETS}"
        )
    download_mode = "force_redownload" if no_cache else "reuse_dataset_if_exists"
    dataset_split = f"{split}[:{sample_size}]" if sample_size else split
    raw = (
        load_dataset(
            dataset_name,
            dataset_config,
            split=dataset_split,
            download_mode=download_mode,
        )
        if dataset_config
        else load_dataset(
            dataset_name, split=dataset_split, download_mode=download_mode
        )
    )
    if shuffle:
        raw = raw.shuffle(seed=seed)

    pairs = []
    for row in raw:
        pair = lang_pairs(row, source_lang, target_lang)
        if pair:
            pairs.append(pair)

    if len(pairs) < val_size + test_size + 1:
        raise ValueError("dataset sample too small for the requested splits")

    train_pairs = pairs[: -(val_size + test_size)]
    val_pairs = pairs[-(val_size + test_size) : -test_size]
    test_pairs = pairs[-test_size:]

    return TranslationData(
        train_loader=_translation_loader(
            train_pairs,
            source_tokenizer,
            target_tokenizer,
            max_length,
            batch_size,
            shuffle=True,
        ),
        val_loader=_translation_loader(
            val_pairs,
            source_tokenizer,
            target_tokenizer,
            max_length,
            batch_size,
            shuffle=False,
        ),
        test_loader=_translation_loader(
            test_pairs,
            source_tokenizer,
            target_tokenizer,
            max_length,
            batch_size,
            shuffle=False,
        ),
        source_tokenizer=source_tokenizer,
        target_tokenizer=target_tokenizer,
        source_lang=source_lang,
        target_lang=target_lang,
    )
