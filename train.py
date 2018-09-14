import torch
import torch.nn as nn 
import torch.optim as optim 
import model
import data


def train(input_bit_strings, output_bit_strings, lr = 1e-1):
    encoder = model.DiffEncoder() 
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(encoder.parameters(), lr = lr)

    for epoch in range(30):

        # Initialize the total loss for each epoch
        total_loss = 0.0

        for input_bits, target_bits in zip(input_bit_strings, output_bit_strings):

            # Reinitialize the state before each sequence, if we put this
            # above the batch loop, we'll get an error about traversing
            # backwards twice through the graph
            state = encoder.init_states()

            # Convert each of the input bit strings accordingly
            input_seq = data.bits2one_hot(input_bits)
            target_seq = data.bits2tensor(target_bits)

            # Start the optimization ...
            optimizer.zero_grad()

            # Update the state
            outputs, state = encoder(input_seq, state)

            # Compute the loss 
            loss = loss_fn(outputs, target_seq)

            # Backpropagate and perform step
            loss.backward()
            optimizer.step()    

            # Keep track of the loss for each epoch
            total_loss += loss.item() 

        # Print out the training error
        print(f'Epoch {epoch}: {total_loss:.6f}')

    # Save the model
    torch.save(encoder, 'diff_encoder.pt')



