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
    
def fit(model, X, y, validation_data=None, epochs=1, batch_size=1, lr=1e-2):
    
    # Initialize the loss function and optimizer
    loss_fn = nn.CrossEntropyLoss() 
    optimizer = optim.Adam(model.parameters(), lr)
    
    model.double()
    
    # One-hot each sequence in X

    
    # Training dataset and data loader
    X = torch.from_numpy(X)
    y = torch.from_numpy(y)
    dataset_train = data.TensorDataset(X, y)
    loader_train = data.DataLoader(dataset_train, batch_size=batch_size)
    
    # Validation dataset and data loader 
    if validation_data is not None:
        Xv, yv = validation_data 
        Xv = torch.from_numpy(Xv)
        yv = torch.from_numpy(yv)
        dataset_valid = data.TensorDataset(Xv, yv)
        loader_validation = data.DataLoader(dataset_valid, batch_size=batch_size)

    # Initialize history
    train_loss = []
    train_acc = []
    validation_loss = []
    validation_acc = []

    # Main loop
    for i in range(epochs):
        
        # Training Loop
        train_loss_e = 0.0
        train_errs_e = 0         
        model.train()
        for input_seqs, target_seqs in loader_train:
            optimizer.zero_grad()
            outputs = model(input_seqs)
            target_bits = target_seqs.reshape(-1)
            loss = loss_fn(outputs, target_bits)
            loss.backward()
            optimizer.step()
            
            train_loss_e += loss.item()
            
            # Evaluate training accuracy
            output_bits = outputs.argmax(dim=1)
            bit_errors = torch.abs(output_bits - target_bits).sum()
            train_errs_e += bit_errors
            
            # Track gradient information to see how learning is progressing

        # Validation Loop
        valid_loss_e = 0.0
        valid_errs_e = 0         
        model.eval()
        for input_seqs, target_seqs in loader_validation:
            outputs = model(input_seqs)
            target_bits = target_seqs.reshape(-1)
            loss = loss_fn(outputs, target_bits)
            
            valid_loss_e += loss.item()
            
            # Evaluate training accuracy
            output_bits = outputs.argmax(dim=1)
            bit_errors = torch.abs(output_bits - target_bits).sum()
            valid_errs_e += bit_errors

        # Validation loop
        if (i % 10 == 0):
            print(f'Epoch {i:2d} .. train-loss: {train_loss_e: 0.4f} train-errs:{train_errs_e: 0.4f} valid-loss: {valid_loss_e: 0.4f} valid-errs: {valid_errs_e: 0.4f}')
            
        train_loss.append(train_loss_e)
        train_acc.append(train_errs_e)
        
    # Build up the history dictionary
    history = {
        'train_loss': train_loss,
        'train_acc': train_acc,
        'validation_acc': validation_acc,
        'validation_loss': validation_loss
    }
    
    return history