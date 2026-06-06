"""Registry

register datasets, models, etc.

Author(s)
---------
Daniel Nicolas Gisolfi <dgisolfi3@gatech.edu>
"""

from typing import Any

from collections.abc import Callable
from dataclasses import dataclass

from belt.data.iris import SupervisedData, iris
from belt.data.translation import TranslationData, load_translation_data

# Data registry
@dataclass(frozen=True)
class DatasetLoaders:
    supervised: Callable[..., SupervisedData] | None = None
    translation: Callable[..., TranslationData] | None = None


dataset_registry: dict[str, DatasetLoaders] = {
    "iris": DatasetLoaders(
        supervised=iris,
    ),
    "opus_books_en_fr": DatasetLoaders(
        translation=load_translation_data,
    ),
    "multi30k_en_de": DatasetLoaders(
        translation=load_translation_data,
    ),
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
