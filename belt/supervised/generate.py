"""Generate Pipeline

Author(s)
---------
Daniel Nicolas Gisolfi <dgisolfi3@gatech.edu>
"""


from belt.supervised.pipeline import SupervisedPipeline


class GeneratePipeline(SupervisedPipeline):
    """Image Classification and Generation"""

    def __init__(self):
        super().__init__()

    def train_batch(self, batch):
        pass

    def eval_batch(self, batch):
        pass

    def score(self, predictions, targets):
        pass


def deploy(config_path, overrides=None):
    return GeneratePipeline().run(config_path, overrides)
