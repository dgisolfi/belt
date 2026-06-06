"""Seq2Seq transformer

Author(s)
---------
Daniel Nicolas Gisolfi <dgisolfi3@gatech.edu>
"""

import torch
from torch import nn

from belt.supervised.models import supervised_model_registry


class Encoder(nn.Module):
    """Reads a sequence of text and compresses it into a fixed context vector

    each source token is embeded and passed through an RNN. The meaning of the source
    is encoded into the projected final hidden layer which is passed to the decoder
    """

    def __init__(
        self,
        input_size,
        emb_size,
        encoder_hidden_size,
        decoder_hidden_size,
        dropout=0.2,
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

        self.L1 = nn.Linear(encoder_hidden_size, encoder_hidden_size)
        self.relu = nn.ReLU()
        self.L2 = nn.Linear(encoder_hidden_size, decoder_hidden_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        # input -> embeddings -> dropout
        embedded = self.dropout(self.emb(input))
        outputs, hidden = self.rnn(embedded)
        # L1 -> ReLU -> L2 -> tanh project into decoder hidden size
        hidden = torch.tanh(self.L2(self.relu(self.L1(hidden))))
        return outputs, hidden


class Decoder(nn.Module):
    """Generate the target sequence a token at a time from the encoder context

    At each step the previous predicted token is embedded and fed into an RNN
    alongside the running hidden state.

    When attention is enabled, cosine similarity scores are computed between the
    current hidden state and all encoder outputs. The weighted context vector is then
    concated with the embedding before the RNN forward pass

    https://arxiv.org/abs/1409.0473
    https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/
    """

    def __init__(
        self,
        emb_size,
        encoder_hidden_size,
        decoder_hidden_size,
        output_size,
        dropout=0.2,
        attention=False,
    ):
        super().__init__()

        self.emb_size = emb_size
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.output_size = output_size
        self.attention = attention

        self.emb = nn.Embedding(output_size, emb_size)

        self.rnn = nn.RNN(
            input_size=emb_size,
            hidden_size=decoder_hidden_size,
            batch_first=True,
        )

        self.linear = nn.Linear(decoder_hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

        self.dropout = nn.Dropout(dropout)
        if self.attention:
            self.attn_proj = nn.Linear(emb_size + encoder_hidden_size, emb_size)

    def compute_attention(self, hidden, encoder_outputs):
        """Compute cosine similarity attention weights over encoder outputs"""
        hidden = hidden.squeeze(0)

        k = encoder_outputs
        q = hidden.unsqueeze(1)

        q_norm = q.norm(dim=2, keepdim=True)
        K_norm = k.norm(dim=2, keepdim=True)

        # cosine similarity = (q T k) / (|q| * |k|)
        sim = torch.bmm(q, k.transpose(1, 2)) / (q_norm * K_norm.transpose(1, 2) + 1e-8)

        # torch.nn.torch.functional.cosine_similarity()
        # compute attention weights from softmax on src pos
        return torch.nn.functional.softmax(sim, dim=-1)

    def forward(self, input, hidden, encoder_outputs=None):
        """Decode one token step"""
        # token id -> embedding -> dropout
        drop = self.dropout(self.emb(input))
        rnn_input = drop

        if encoder_outputs is not None and self.attention:
            # build the context vec with the weighted encoder outputs by sim to the hidden
            attn_weights = self.compute_attention(hidden, encoder_outputs)
            context = torch.bmm(attn_weights, encoder_outputs)
            # concat with embedding to poject to emb_size
            rnn_input = self.attn_proj(torch.cat([context, drop], dim=2))

        output, hidden = self.rnn(rnn_input, hidden)
        # vocab log-softmax probabilities
        output = self.softmax(self.linear(output.squeeze(1)))

        return output, hidden


@supervised_model_registry.register("Seq2Seq")
class Seq2Seq(nn.Module):
    """Dual RNN Encoder-Decoder model for sequence-to-sequence tasks

    The Encoder consumes the source sequence and produces a context vector,
    as the Decoder generates the target sequence autoregressively. The
    predicted token is fed as input to the next decoder step (greedy decoding)

    https://arxiv.org/abs/1409.3215
    """

    def __init__(
        self,
        source_vocab_size,
        target_vocab_size,
        emb_size,
        encoder_hidden_size,
        decoder_hidden_size,
        device,
        dropout=0.3,
        attention=False,
        **kwargs,
    ):
        super().__init__()
        self.device = device
        self.encoder = Encoder(
            input_size=source_vocab_size,
            emb_size=emb_size,
            encoder_hidden_size=encoder_hidden_size,
            decoder_hidden_size=decoder_hidden_size,
            dropout=dropout,
        ).to(device)
        self.decoder = Decoder(
            emb_size=emb_size,
            encoder_hidden_size=encoder_hidden_size,
            decoder_hidden_size=decoder_hidden_size,
            output_size=target_vocab_size,
            dropout=dropout,
            attention=attention,
        ).to(device)

    def forward(self, source):
        """Run autoregressive decoding for a source token batch."""
        batch_size = source.shape[0]
        seq_len = source.shape[1]

        outputs = torch.zeros(
            batch_size, seq_len, self.decoder.output_size, device=self.device
        )
        # add intial src token in decoder
        decoder_input = source[:, 0].unsqueeze(1)

        e_out, hidden = self.encoder.forward(source)

        for time in range(seq_len):
            d_out, hidden = self.decoder.forward(decoder_input, hidden, e_out)
            # save log probabilities
            decoder_input = d_out.argmax(1).unsqueeze(1)
            outputs[:, time, :] = d_out
        return outputs
