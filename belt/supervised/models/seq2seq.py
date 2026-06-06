"""Seq2Seq transformer

Author(s)
---------
Daniel Nicolas Gisolfi <dgisolfi3@gatech.edu>
"""

from torch import nn


class Encoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Encoder).__init__(*args, **kwargs)


class Decoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Decoder).__init__(*args, **kwargs)


class Seq2Seq(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Seq2Seq).__init__(*args, **kwargs)
