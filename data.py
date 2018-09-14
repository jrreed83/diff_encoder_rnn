import torch
import random

def bits2tensor(bit_string):
    # Assumes that the left-most character is the most-significant bit,
    # hence the reversal
    l = [int(c) for c in bit_string]
    return torch.tensor(l)

def bits2one_hot(bit_string):
    seq_len = len(bit_string)
    l = [int(c) for c in bit_string]

    # We're using a batch size of 1
    output = torch.zeros(seq_len, 1, 2)
    for i, li in enumerate(l):
        output[i, :, li] = 1.0
    return output

def generate_training_data(num_strings = 1000, size = 10):
    # Generate a bunch of random input bit strings
    inputs = []
    outputs = []

    for i in range(num_strings):
        xi = ''.join([random.choice(['0','1']) for j in range(size)])
        yi = diff_encode(xi)

        inputs.append(xi)
        outputs.append(yi)
    # Differentially encode all of them

    return inputs, outputs

def diff_encode(bit_string):
    bits_i = [int(ci) for ci in bit_string]
    bits_o = [0] * len(bit_string)

    state = 0
    for i, xi in enumerate(bits_i):
        yi = state ^ xi 
        state = yi
        bits_o[i] = yi

    output = ''.join([str(yi) for yi in bits_o])
    return output

