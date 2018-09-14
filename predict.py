import torch 
import data 
def predict():
    encoder = torch.load('diff_encoder.pt')

    # Let's run a simple prediction ...
    input_seq = data.bits2one_hot('10101010001010')    
    state = encoder.init_states()
    out, _ = encoder(input_seq, state)
    y = torch.argmax(out, dim=1).numpy()
    y = ''.join([str(yi) for yi in y])
    return y 