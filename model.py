import torch
import torch.nn as nn 

class DiffEncoder(nn.Module):
    def __init__(self, state_dim = 4):
        super().__init__()
        
        # We assume here that the input and output
        # bits are one-hot-encoded
        self.state_dim = state_dim
        self.rnn_cell = nn.GRUCell(2, state_dim)
        self.linear = nn.Linear(state_dim, 2)

    def forward(self, input_seq, state):

        # Because we're using the RNN Cell, input_seq would
        # ordinarily be of size 
        # (sequence_length, batch_size, vocab_length), 
        # however, I'm choosing not to batch anything.
        seq_len, _, _ = input_seq.size()

        output_seq = torch.zeros((seq_len, 2))

        # Iterate over the entire input sequence, updating
        # the internal state of the cell as we go and
        # generating our output sequence       
        for i, input in enumerate(input_seq):
            state = self.rnn_cell(input, state)  

            # Apply the linear layer
            output_seq[i] = self.linear(state) 

        return output_seq, state 

    def init_states(self):
        return torch.zeros(1, self.state_dim)
