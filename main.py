import torch
import model as M 
import numpy as np 
import matplotlib.pyplot as plt 

def diff_encode(x, h=0):
    y = np.zeros(len(x), dtype=np.int)
    for i, xi in enumerate(x):
        # Compute output
        yi = xi ^ h
        # Update state for next iteration 
        h = yi
        # Store output
        y[i] = yi
    return y

def encode_seqs(bit_sequences):
    num_seqs, seq_len = bit_sequences.shape
    y = np.zeros((num_seqs, seq_len), dtype=np.int)
    for i, seq in enumerate(bit_sequences):
        y[i,:] = diff_encode(seq)
    return y   

def one_hot_encoding(bit_sequences):
    '''
    One-hot encode a sequence of bits
    '''
    num_seqs, seq_len = bit_sequences.shape
    x = np.zeros((num_seqs, seq_len, 2), dtype=np.double)
    for i, seq in enumerate(bit_sequences):
        for j, bit in enumerate(seq):
            if bit == 0:
                x[i,j,:] = [1, 0]
            elif bit == 1:
                x[i,j,:] = [0, 1]
    return x
def constellation(bit_sequences):
    '''
    One-hot encode a sequence of bits
    '''
    num_seqs, seq_len = bit_sequences.shape
    x = np.zeros((num_seqs, seq_len, 2), dtype=np.double)
    for i, seq in enumerate(bit_sequences):
        for j, bit in enumerate(seq):
            if bit == 0:
                x[i,j,:] = [-1, 0]
            elif bit == 1:
                x[i,j,:] = [+1, 0]
    return x
def main():
    model = M.Network()

    training_msgs = np.array([
        [0,0,0],
        [1,1,1],
        [0,1,0],
        [1,0,1],
        [1,1,0],
        [0,0,1],
        [0,1,1],
        [1,0,0] 
    ])
    X = one_hot_encoding(training_msgs.copy())
    y = encode_seqs(training_msgs.copy())    

    validation_msgs = np.array([
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [1,0,0,1,0,0,1,1,1,0,0,1,1,0],
        [0,1,1,0,0,0,1,1,1,1,0,0,0,0],
        [1,1,0,0,1,1,0,0,1,1,0,0,1,1]
    ])
    Xv = one_hot_encoding(validation_msgs.copy())
    yv = encode_seqs(validation_msgs.copy())  

    history = M.fit(model, X, y, validation_data=(Xv, yv), epochs=300)

if __name__ == '__main__':
    main()

