import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from dataset import Shakespeare
from model import CharRNN, CharLSTM
import matplotlib.pyplot as plt

def train(model, trn_loader, device, criterion, optimizer):
    """ Train function

    Args:
        model: network
        trn_loader: torch.utils.data.DataLoader instance for training
        device: device for computing, cpu or gpu
        criterion: cost function
        optimizer: optimization method, refer to torch.optim

    Returns:
        trn_loss: average loss value
    """

    model.train()
    trn_loss = 0

    for i, (inputs, targets) in enumerate(trn_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        hidden = model.init_hidden(inputs.size(0))
        if isinstance(hidden, tuple):
            hidden = tuple(h.to(device) for h in hidden)
        else:
            hidden = hidden.to(device)

        outputs, _ = model(inputs, hidden)

        loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        trn_loss += loss.item()

    trn_loss /= len(trn_loader)
    return trn_loss

def validate(model, val_loader, device, criterion):
    """ Validate function

    Args:
        model: network
        val_loader: torch.utils.data.DataLoader instance for testing
        device: device for computing, cpu or gpu
        criterion: cost function

    Returns:
        val_loss: average loss value
    """

    model.eval()
    val_loss = 0

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            hidden = model.init_hidden(inputs.size(0))
            if isinstance(hidden, tuple):
                hidden = tuple(h.to(device) for h in hidden)
            else:
                hidden = hidden.to(device)

            outputs, _ = model(inputs, hidden)

            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            val_loss += loss.item()

    val_loss /= len(val_loader)
    return val_loss


def main(model_type):
    """ Main function

        Here, you should instantiate
        1) DataLoaders for training and validation. 
           Try SubsetRandomSampler to create these DataLoaders.
        3) model
        4) optimizer
        5) cost function: use torch.nn.CrossEntropyLoss

    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_file = 'shakespeare_train.txt'

    dataset = Shakespeare(input_file)
    dataset_size = len(dataset)
    indices = list(range(dataset_size))

    train_split = 0.8
    train_size = int(train_split * dataset_size)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    batch_size = 64
    trn_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)

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

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    num_epochs = 50
    patience = 5
    best_val_loss = float('inf')
    counter = 0
    early_stop_epoch = num_epochs

    trn_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        trn_loss = train(model, trn_loader, device, criterion, optimizer)
        val_loss = validate(model, val_loader, device, criterion)

        trn_losses.append(trn_loss)
        val_losses.append(val_loss)

        print(f"[{model_type}] Epoch [{epoch+1}/{num_epochs}], Train Loss: {trn_loss:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                early_stop_epoch = epoch + 1
                print(f"Early stopping at epoch {early_stop_epoch}")
                break

    torch.save(model.state_dict(), f"{model_type}_model.pt")
    print(f"{model_type} model saved as {model_type}_model.pt")

    return trn_losses, val_losses, early_stop_epoch

if __name__ == '__main__':
    rnn_trn_losses, rnn_val_losses, rnn_early_stop_epoch = main('RNN')
    lstm_trn_losses, lstm_val_losses, lstm_early_stop_epoch = main('LSTM')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    epochs = range(1, len(rnn_trn_losses) + 1)
    ax1.plot(epochs, rnn_trn_losses, label='RNN Training Loss')
    ax1.plot(epochs, rnn_val_losses, label='RNN Validation Loss')
    ax1.axvline(rnn_early_stop_epoch, color='r', linestyle='--', label=f'Early Stopping (Epoch {rnn_early_stop_epoch})')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('RNN Loss Curve')
    ax1.set_xticks(epochs)
    ax1.legend()

    epochs = range(1, len(lstm_trn_losses) + 1)
    ax2.plot(epochs, lstm_trn_losses, label='LSTM Training Loss')
    ax2.plot(epochs, lstm_val_losses, label='LSTM Validation Loss')
    ax2.axvline(lstm_early_stop_epoch, color='r', linestyle='--', label=f'Early Stopping (Epoch {lstm_early_stop_epoch})')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('LSTM Loss Curve')
    ax2.set_xticks(epochs)
    ax2.legend()

    plt.tight_layout()
    plt.savefig('rnn_vs_lstm_loss_curve_with_early_stopping.png')
    plt.show()