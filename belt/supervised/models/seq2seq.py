"""Seq2Seq transformer

Author(s)
---------
Daniel Nicolas Gisolfi <dgisolfi3@gatech.edu>
"""
import torch
from torch import nn


class Encoder(nn.Module):
    """Encoder module for the Seq2Seq model"""
    def __init__(
        self, 
        input_size,
        emb_size,
        encoder_hidden_size,
        decoder_hidden_size,
        dropout=0.2
    ):
        super().__init__()
        self.input_size = input_size
        self.emb_size = emb_size
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size

        self.emb = nn.Embedding(input_size, emb_size)
        self.rnn = nn.RNN(
            input_size=emb_size,
            hidden_size=encoder_hidden_size,
            batch_first=True,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        """Encode source token IDs"""
        embeddings = self.emb(input)

class Decoder(nn.Module):
    """Decoder module for the Seq2Seq model"""

    def __init__(self,
        emb_size,
        encoder_hidden_size,
        decoder_hidden_size,
        output_size,
        dropout=0.2
    ):
        super().__init__()
        
        self.emb_size = emb_size
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.output_size = output_size

        self.emb = nn.Embedding(output_size, emb_size)
        self.rnn = nn.RNN(
            input_size=emb_size,
            hidden_size=decoder_hidden_size,
            batch_first=True,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden):
        """Decode one token step"""
        embeddings = self.emb(input)

class Seq2Seq(nn.Module):
    """Sequence-to-sequence encoder/decoder model"""

    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.device = device
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)

    def forward(self, source):
        """autoregressive decoding per source token batch"""