import torch
import torch.nn as nn

class RNNUnit(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RNNUnit, self).__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_linear = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x, h):
        h_next = torch.tanh(self.input_layer(x) + self.hidden_linear(h))
        return h_next


class LSTMUnit(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()

        combined_size = input_size + hidden_size

        self.input_gate = nn.Linear(combined_size, hidden_size)
        self.forget_gate = nn.Linear(combined_size, hidden_size)
        self.candidate_layer = nn.Linear(combined_size, hidden_size)
        self.output_gate = nn.Linear(combined_size, hidden_size)

        # Initialize forget gate bias to 1 for better training stability
        nn.init.ones_(self.forget_gate.bias)

    def forward(self, x, h, c):
        combined = torch.cat([x, h], dim=-1)

        i = torch.sigmoid(self.input_gate(combined))
        f = torch.sigmoid(self.forget_gate(combined))
        g = torch.tanh(self.candidate_layer(combined))
        o = torch.sigmoid(self.output_gate(combined))

        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next
    
class RNN(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim=32,
        hidden_size=64,
        num_layers=2
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.rnn_units = nn.ModuleList([
            RNNUnit(
                embedding_dim if i == 0 else hidden_size,
                hidden_size
            )
            for i in range(num_layers)
        ])
        self.output_layer = nn.Linear(hidden_size, vocab_size)

    def forward(self, token_ids, state=None):
        x = self.embedding(token_ids)

        batch_size, seq_len, _ = x.shape

        if state is None:
            h = [
                x.new_zeros(batch_size, self.hidden_size)
                for _ in range(self.num_layers)
            ]
        else:
            h = list(state.unbind(dim=0))

        outputs = []

        for t in range(seq_len):
            layer_input = x[:, t, :]

            for i, rnn_unit in enumerate(self.rnn_units):
                h[i] = rnn_unit(layer_input, h[i])
                layer_input = h[i]

            outputs.append(layer_input)

        output = torch.stack(outputs, dim=1)

        logits = self.output_layer(output)
        h_n = torch.stack(h, dim=0)

        return logits, h_n

class LSTM(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim=32,
        hidden_size=64,
        num_layers=2
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.lstm_units = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                input_size = embedding_dim
            else:
                input_size = hidden_size

            self.lstm_units.append(LSTMUnit(input_size, hidden_size))

        self.output_layer = nn.Linear(hidden_size, vocab_size)

    def forward(self, token_ids, state=None):
        x = self.embedding(token_ids)

        batch_size, seq_len, _ = x.shape

        if state is None:
            h = [
                x.new_zeros(batch_size, self.hidden_size)
                for _ in range(self.num_layers)
            ]

            c = [
                x.new_zeros(batch_size, self.hidden_size)
                for _ in range(self.num_layers)
            ]
        else:
            h_0, c_0 = state
            h = list(h_0.unbind(dim=0))
            c = list(c_0.unbind(dim=0))

        outputs = []

        for t in range(seq_len):
            layer_input = x[:, t, :]

            for i, lstm_unit in enumerate(self.lstm_units):
                h[i], c[i] = lstm_unit(
                    layer_input, h[i], c[i]
                )

                layer_input = h[i]

            outputs.append(layer_input)

        output = torch.stack(outputs, dim=1)

        logits = self.output_layer(output)

        h_n = torch.stack(h, dim=0)
        c_n = torch.stack(c, dim=0)

        return logits, (h_n, c_n)
