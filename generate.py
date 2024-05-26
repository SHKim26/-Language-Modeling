import torch
from model import CharRNN, CharLSTM
from dataset import Shakespeare

def generate(model, dataset, seed_characters, temperature, device, seq_length=100):
    """ Generate characters

    Args:
        model: trained model
        dataset: Shakespeare dataset instance
        seed_characters: seed characters
        temperature: T
        device: device for computing, cpu or gpu
        seq_length: number of characters to generate

    Returns:
        samples: generated characters
    """

    model.eval()
    generated_chars = seed_characters

    # Convert seed characters to indices
    seed_indices = [dataset.char_to_idx[c] for c in seed_characters]
    input_seq = torch.tensor(seed_indices, dtype=torch.long).unsqueeze(0).to(device)

    hidden = model.init_hidden(1)
    if isinstance(hidden, tuple):
        hidden = tuple(h.to(device) for h in hidden)
    else:
        hidden = hidden.to(device)

    for _ in range(seq_length):
        output, hidden = model(input_seq, hidden)
        
        # Apply temperature
        output = output[-1] / temperature
        probs = torch.softmax(output, dim=-1)
        
        # Sample from the distribution
        char_index = torch.multinomial(probs, 1).item()
        
        # Append the new character to the generated sequence
        generated_chars += dataset.idx_to_char[char_index]
        
        # Use the new character as the input for the next step
        input_seq = torch.tensor([[char_index]], dtype=torch.long).to(device)

    return generated_chars

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_file = 'shakespeare_train.txt'
    dataset = Shakespeare(input_file)

    # Check RNN or LSTM model performance
    model_type = 'LSTM'  

    input_size = len(dataset.chars)
    hidden_size = 256
    output_size = input_size
    num_layers = 2

    if model_type == 'RNN':
        model = CharRNN(input_size, hidden_size, output_size, num_layers).to(device)
    elif model_type == 'LSTM':
        model = CharLSTM(input_size, hidden_size, output_size, num_layers).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.load_state_dict(torch.load(f"{model_type}_model.pt"))

    num_samples = 5
    seq_length = 100

    for temperature in [0.5, 1.0, 2.0]:
        with open(f"{model_type}_generated_temp_{temperature}.txt", "w", encoding="utf-8") as f:
            f.write(f"Temperature: {temperature}\n\n")
            for i in range(num_samples):
                seed_characters = dataset.data[torch.randint(len(dataset.data) - dataset.seq_length, size=(1,)).item()][:dataset.seq_length]
                generated_chars = generate(model, dataset, seed_characters, temperature, device, seq_length)
                f.write(f"Sample {i+1}:\n")
                f.write(f"Seed: {seed_characters}\n")
                f.write(f"Generated: {generated_chars}\n\n")