"""Iris Dataset

basic dataset for validating models

Author(s)
---------
Daniel Nicolas Gisolfi <dgisolfi3@gatech.edu>
"""

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class SupervisedData:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    class_names: list[str]

def load_dataset(
    features: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    shuffle: bool,
):
    dataset = TensorDataset(
        torch.tensor(features, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.long),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def iris(
    test_size: float,
    val_size: float,
    batch_size: int,
    seed: int,
):
    iris = load_iris()
    features = iris.data.astype("float32")
    labels = iris.target.astype("int64")
    class_names = list(iris.target_names)
    # test set
    x_train_val, x_test, y_train_val, y_test = train_test_split(
        features,
        labels,
        test_size=test_size,
        random_state=seed,
        stratify=labels,
    )
    # remaining portion of dataset is for validation
    val_fraction = val_size / (1.0 - test_size)
    # split reamining train to create validation set
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val,
        y_train_val,
        test_size=val_fraction,
        random_state=seed,
        stratify=y_train_val,
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train).astype("float32")
    x_val = scaler.transform(x_val).astype("float32")
    x_test = scaler.transform(x_test).astype("float32")

    return SupervisedData(
        train_loader=load_dataset(x_train, y_train, batch_size=batch_size, shuffle=True),
        val_loader=load_dataset(x_val, y_val, batch_size=batch_size, shuffle=False),
        test_loader=load_dataset(x_test, y_test, batch_size=batch_size, shuffle=False),
        class_names=class_names,
    )


