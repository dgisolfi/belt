"""Image Datasets

- CIFAR-10
- Fashion-MNIST

Author(s)
---------
Daniel Nicolas Gisolfi <dgisolfi3@gatech.edu>
"""


from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, FashionMNIST

from belt.data.iris import SupervisedData
from belt.utils import DATA_DIR

# https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html
fashion_mnist_class_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
CLASS_NAMES = fashion_mnist_class_map.values()


def load_fashion_mnist(batch_size=64, **kwargs):
    train = FashionMNIST(root=DATA_DIR, train=True, download=True)
    test = FashionMNIST(root=DATA_DIR, train=False, download=True)

    return SupervisedData(
        train_loader=DataLoader(train, batch_size=batch_size, shuffle=True),
        val_loader=DataLoader(test, batch_size=batch_size, shuffle=False),
        test_loader=DataLoader(test, batch_size=batch_size, shuffle=False),
        class_names=CLASS_NAMES,
    )


def load_cifar10(batch_size=64, **kwargs):
    train = CIFAR10(root=DATA_DIR, train=True, download=True)
    test = CIFAR10(root=DATA_DIR, train=False, download=True)

    return SupervisedData(
        train_loader=DataLoader(train, batch_size=batch_size, shuffle=True),
        val_loader=DataLoader(test, batch_size=batch_size, shuffle=False),
        test_loader=DataLoader(test, batch_size=batch_size, shuffle=False),
        class_names=CLASS_NAMES,
    )
