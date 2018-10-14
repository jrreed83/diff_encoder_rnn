import torch
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data

class Network(nn.Module):
    def __init__(self, n_hidden=2):
        super().__init__()

        self.n_hidden = n_hidden
        self.rnn = nn.GRU(2, self.n_hidden, batch_first=True)
        self.lin = nn.Linear(self.n_hidden, 2)
        
    def forward(self, sequences):
        output, _ = self.rnn(sequences)
        output = output.reshape(-1, self.n_hidden)
        output = self.lin(output)
        return output  
    
def fit(model, X, y, epochs=1, batch_size=1, lr=1e-2):
    
    # Initialize the loss function and optimizer
    loss_fn = nn.CrossEntropyLoss() 
    optimizer = optim.Adam(model.parameters(), lr)
    
    model.double()
    
    # One-hot each sequence in X
    X = torch.from_numpy(X)
    y = torch.from_numpy(y)
    
    # Training dataset and data loader
    dataset_train = data.TensorDataset(X, y)
    loader_train = data.DataLoader(dataset_train, batch_size=batch_size)
    
    # Validation dataset and data loader 
    
    # Initialize history
    train_loss = []
    train_acc = []
    
    # Main loop
    for i in range(epochs):
        
        # Training Loop
        epoch_loss = 0.0
        epoch_acc = 0
        model.train()
        for input_seqs, target_seqs in loader_train:
            optimizer.zero_grad()
            outputs = model(input_seqs)
            target_bits = target_seqs.reshape(-1)
            loss = loss_fn(outputs, target_bits)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Evaluate training accuracy
            output_bits = outputs.argmax(dim=1)
            bit_errors = torch.abs(output_bits - target_bits).sum()
            epoch_acc += bit_errors
            
            # Track gradient information to see how learning is progressing
            
        # Validation loop
        if (i % 10 == 0):
            print(f'Epoch {i:2d} ........ loss: {epoch_loss: 0.4f} - acc: {epoch_acc: 0.4f}')
            
        train_loss.append(epoch_loss)
        train_acc.append(epoch_acc)
        
    # Build up the history dictionary
    history = {
        'train_loss': train_loss,
        'train_acc': train_acc
    }
    
    return history