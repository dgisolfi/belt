"""Registry

register datasets, models, etc.

Author(s)
---------
Daniel Nicolas Gisolfi <dgisolfi3@gatech.edu>
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from belt.data.iris import SupervisedData, iris


# Data registry
@dataclass(frozen=True)
class DatasetLoaders:
    supervised: Callable[..., SupervisedData]

dataset_registry: dict[str, DatasetLoaders] = {
    "iris": DatasetLoaders(
        supervised=iris,
    )
}

# Basic Registry for Models and other composable components
class Registry(dict[str, type]):
    """Maps string names to classes and supports decorator-based registration."""

    def register(self, name: str):
        def decorator(cls: type) -> type:
            if name in self:
                raise KeyError(f"'{name}' is already registered in this registry")
            self[name] = cls
            return cls

        return decorator

    def build(self, name: str, **kwargs: Any) -> Any:
        if name not in self:
            raise KeyError(f"'{name}' not found. Available: {sorted(self.keys())}")
        return self[name](**kwargs)
