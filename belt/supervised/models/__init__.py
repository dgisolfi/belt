from belt.registry import Registry

supervised_model_registry: Registry = Registry()

from belt.supervised.models import (seq2seq, softmax_regressor)
