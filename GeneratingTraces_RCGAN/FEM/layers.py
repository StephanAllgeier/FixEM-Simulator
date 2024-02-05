import torch.nn as nn


def rnn_layer(input_size,
              hidden_size=None,
              num_layers=1,
              dropout=0.5,
              rnn_type='rnn',
              nonlinearity='relu'):
    # Set hidden_size to input_size if not specified
    hidden_size = hidden_size or input_size

    rnn_types = {
        'rnn': nn.RNN,
        'lstm': nn.LSTM,
        'gru': nn.GRU
    }

    rnn_kwargs = dict(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        batch_first=True,
        dropout=dropout
    )

    if rnn_type == 'rnn':
        rnn_kwargs['nonlinearity'] = nonlinearity

    return rnn_types[rnn_type](**rnn_kwargs)
