import random

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from ignite.metrics import Rouge
from torcheval.metrics.functional import bleu_score
from tqdm import tqdm

from utils.models import AttnDecoderRNN, EncoderRNN

plt.switch_backend('agg')
import argparse
import os
import pickle

import numpy as np


def plot_metrics(figures_dir, train_losses, val_losses, val_metrics):
    """
    Plots the training and validation losses.
    """
    plt.rcParams.update({'font.size': 22})
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    # Plot training and validation losses
    ax[0].plot(train_losses, label='Training Loss')
    ax[0].plot(val_losses, label='Validation Loss')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss')
    ax[0].set_title('Training and Validation Losses')
    ax[0].legend()

    # Plot validation metrics
    for metric_name, metric_values in val_metrics.items():
        ax[1].plot(metric_values, label=metric_name)
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Metric Value')
    ax[1].set_title('Validation Metrics')
    ax[1].legend()
    plt.savefig(os.path.join(figures_dir, 'metrics.png'))
    plt.close(fig)

def evaluate_loss(dataloader, encoder, decoder, criterion):
    total_loss = 0
    with torch.no_grad():
        for data in tqdm(dataloader):
            input_tensor, target_tensor = data

            encoder_outputs, encoder_hidden = encoder(input_tensor)
            decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_tensor)

            loss = criterion(
            decoder_outputs.view(-1, decoder_outputs.size(-1)),
            target_tensor.view(-1)
            )
            total_loss += loss.item()

    return total_loss / len(dataloader)

def decode_data(text_ids, index2word, EOS_token):
    """
    Converts the text ids to words using the index2word mapping.
    """
    if text_ids.dim() > 1:
        text_ids = text_ids.view(-1)  # Flatten to 1D

    decoded_words = []
    for idx in text_ids:
        # Ensure idx is a scalar
        if isinstance(idx, torch.Tensor):
            idx = idx.item()
        if idx == EOS_token:
            decoded_words.append('EOS')
            break
        decoded_words.append(index2word.get(idx, 'UNK'))

    return " ".join(decoded_words)

def make_predictions(encoder, decoder, input_tensor, index2word, EOS_token):
    """
    Computes the summary for the given input tensor.
    """
    input_tensor = input_tensor[0].unsqueeze(0)
    target_tensor = None  # Set target_tensor to None for inference

    encoder_outputs, encoder_hidden = encoder(input_tensor)
    decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_tensor)

    # Get the predicted words
    _, topi = decoder_outputs.topk(1)
    decoded_words = decode_data(topi.squeeze(), index2word, EOS_token)

    return decoded_words

def compute_metrics(predictions, targets, n1=1, n2=2):
    """
    Computes the BLEU score for n1-grams and ROUGE-n1, ROGUE-n2 and ROGUE-L score for the predictions and targets.
    """    
    metrics = {}
    rouge_metrics = Rouge(variants=["L", n1, n2], multiref="best")
    
    metrics["bleu"] = bleu_score(predictions, targets, n_gram=n1)

    list_predictions = []
    list_targets = []
    for pred in predictions:
        list_predictions.append(pred.split())
    for target in targets:
        list_targets.append(target.split())

    rouge_metrics.update((list_predictions,  [list_targets]))
    metrics.update(rouge_metrics.compute())

    return metrics

def evaluate_model(encoder, decoder, dataloader, index2word, EOS_token):
    """
    Evaluates the model on the given dataloader.
    """
    encoder.eval()
    decoder.eval()

    predictions = []
    targets = []
    
    with torch.no_grad():
        for data in tqdm(dataloader):
            input_tensor, target_tensor = data

            predicted_words = make_predictions(encoder, decoder, input_tensor, index2word, EOS_token)
            target_words = decode_data(target_tensor[0], index2word, EOS_token)

            predictions.append(predicted_words)
            targets.append(target_words)

    return compute_metrics(predictions, targets, n1=1, n2=2)

def train_epoch(dataloader, encoder, decoder, encoder_optimizer,
          decoder_optimizer, criterion):

    encoder.train()
    decoder.train()

    total_loss = 0
    for data in tqdm(dataloader):
        input_tensor, target_tensor = data

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_tensor)

        loss = criterion(
            decoder_outputs.view(-1, decoder_outputs.size(-1)),
            target_tensor.view(-1)
        )
        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def train(train_dataloader, val_dataloader, encoder, decoder, criterion,
          index2words, EOS_token, save_directory, figures_dir, optimizer_hyperparams,
          print_hyperparams):
    
    learning_rate = optimizer_hyperparams['learning_rate']
    weight_decay = optimizer_hyperparams['weight_decay']
    n_epochs = optimizer_hyperparams['n_epochs']
    
    print_every = print_hyperparams['print_every']
    plot_every = print_hyperparams['plot_every']
    print_examples_every = print_hyperparams['print_examples_every']

    # Initializations
    print('Initializing ...')
    plot_train_losses = []
    plot_val_losses = []
    plot_val_metrics = {"BLEU": [], "Rouge-L-F": [], "Rouge-1-F": [], "Rouge-2-F": []}
    print_train_loss_total = 0  # Reset every print_every
    plot_train_loss_total = 0  # Reset every plot_every
    print_val_loss_total = 0  # Reset every print_every
    plot_val_loss_total = 0  # Reset every plot_every
    best_val_loss = float('inf')

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate, weight_decay=weight_decay)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Training loop
    print("Training...")
    for epoch in range(1, n_epochs + 1):
        training_loss = train_epoch(train_dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_train_loss_total += training_loss
        plot_train_loss_total += training_loss

        # Evaluate on validation set
        val_loss = evaluate_loss(val_dataloader, encoder, decoder, criterion)
        print_val_loss_total += val_loss
        plot_val_loss_total += val_loss

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'en': encoder.state_dict(),
                'de': decoder.state_dict(),
            }, os.path.join(save_directory, 'best_checkpoint.tar'))

        # Print progress
        if epoch % print_every == 0:
            print_train_loss_total = print_train_loss_total / print_every
            print_val_loss_total = print_val_loss_total / print_every
            print('epoch: {}; Average training loss: {:.4f}; Average validation loss: {:.4f}.'.format(
                    epoch, epoch / n_epochs * 100, print_train_loss_total, print_val_loss_total))

        if epoch % print_examples_every == 0:
            # Compute metrics for validation set
            val_metrics = evaluate_model(encoder, decoder, val_dataloader, index2words, EOS_token)
            print('-----------------------------------')
            for key in plot_val_metrics.keys():
                plot_val_metrics[key].append(val_metrics[key])
                print('{}: {:.4f}'.format(f"{key} score", val_metrics[key]))
            print('-----------------------------------')
    
            print_train_loss_total = 0
            print_val_loss_total = 0
            
            # Get a random sample from the validation set
            nb_decoding_test = 5
            count_test = 0
            random_list = random.sample(range(len(train_dataloader)), nb_decoding_test)
            for i, data in enumerate(train_dataloader):
                if i in random_list:
                    input_tensor, target_tensor = data
                    print('Input: {}'.format(decode_data(input_tensor[0], index2words, EOS_token)))
                    print('Target: {}'.format(decode_data(target_tensor[0], index2words, EOS_token)))
                    print('-----------------------------------')
                    count_test += 1
                if count_test == nb_decoding_test:
                    break
            print('-----------------------------------')

        # Plot loss progress
        if epoch % plot_every == 0:
            plot_loss_avg = plot_train_loss_total / plot_every
            plot_train_losses.append(plot_loss_avg)
            plot_val_loss_avg = plot_val_loss_total / plot_every
            plot_val_losses.append(plot_val_loss_avg)
            plot_train_loss_total = 0
            plot_val_loss_total = 0

    plot_metrics(figures_dir, plot_train_losses, plot_val_losses, plot_val_metrics)

def main(root_dir,
    model_hyperparams,
    optimizer_hyperparams,
    print_hyperparams,
    load_checkpoint = False,
    name = "WikiHow",
    ):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Random seed for reproducibility
    random.seed(5719)
    np.random.seed(5719)
    torch.manual_seed(5719)
    torch.use_deterministic_algorithms(False)

    # Get directories
    dataset_dir = os.path.join(root_dir, 'data', name)
    save_dir = os.path.join(root_dir, 'checkpoints', name)
    figures_dir = os.path.join(root_dir, 'figures', name)

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    
    # Get the hyperparameters
    batch_size = optimizer_hyperparams['batch_size']
    # num_workers = optimizer_hyperparams['num_workers']
    max_length = model_hyperparams['max_length']
    hidden_size = model_hyperparams['hidden_size']

    # Load the dataset
    X_train = torch.load(os.path.join(dataset_dir, "x_train.pt"))
    X_val = torch.load(os.path.join(dataset_dir, "x_val.pt"))
    X_test = torch.load(os.path.join(dataset_dir, "x_test.pt"))
    y_train = torch.load(os.path.join(dataset_dir, "y_train.pt"))
    y_val = torch.load(os.path.join(dataset_dir, "y_val.pt"))
    y_test = torch.load(os.path.join(dataset_dir, "y_test.pt"))

    train_dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True,
    )
    val_dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_val, y_val),
        batch_size=batch_size,
        shuffle=True,
    )
    test_dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_test, y_test),
        batch_size=batch_size,
        shuffle=True,
    )

    # Load the vocabulary
    print("Loading tokenizer")
    with open(os.path.join(dataset_dir, 'feature_tokenizer.pickle'), 'rb') as handle:
            feature_tokenizer = pickle.load(handle)

    num_words_text = max(feature_tokenizer.word2index.values()) + 1
    EOS_token = feature_tokenizer.word2index.get("EOS", 2)

    # Initialize the model
    print("Initialize the model")
    encoder = EncoderRNN(num_words_text, hidden_size).to(device)
    decoder = AttnDecoderRNN(hidden_size, num_words_text, max_length).to(device)
    criterion = nn.NLLLoss(ignore_index=0)

    if load_checkpoint:
      # Load the best model
      checkpoint = torch.load(os.path.join(save_dir, 'best_checkpoint.tar'))
      encoder.load_state_dict(checkpoint['encoder'])
      decoder.load_state_dict(checkpoint['decoder'])

    # Train the model
    train(train_dataloader, val_dataloader, encoder, decoder, criterion,
          feature_tokenizer.index2word, EOS_token, save_dir, figures_dir,
          optimizer_hyperparams, print_hyperparams)

    # Load the best model
    checkpoint = torch.load(os.path.join(save_dir, 'best_checkpoint.tar'))
    encoder.load_state_dict(checkpoint['en'])
    decoder.load_state_dict(checkpoint['de'])

    # Test the model
    test_loss = evaluate_loss(test_dataloader, encoder, decoder, criterion)
    print('Test loss: {:.4f}'.format(test_loss))

    # Evaluate the model
    metrics = evaluate_model(encoder, decoder, test_dataloader, feature_tokenizer.index2word, EOS_token)
    print('-----------------------------------')
    for key in metrics.keys():
        print('{}: {:.4f}'.format(f"{key} score", metrics[key]))
    print('-----------------------------------')
    # Get a random sample from the test set
    index = random.randint(0, len(test_dataloader) - 1)
    for i, data in enumerate(test_dataloader):
        if i == index:
            input_tensor, target_tensor = data
            break
    decoded_words, = make_predictions(encoder, decoder, input_tensor, feature_tokenizer.index2word, EOS_token)
    print('Input: {}'.format(decode_data(input_tensor[0], feature_tokenizer.index2word, EOS_token)))
    print('Target: {}'.format(decode_data(target_tensor[0], feature_tokenizer.index2word, EOS_token)))
    print('Predicted: {}'.format(decoded_words))
    print('-----------------------------------')

if __name__ == "__main__":
    # Argparse command line arguments
    parser = argparse.ArgumentParser(description='Train a Seq2Seq model with attention.')
    parser.add_argument('--name', type=str, default="WikiHow", help='Name of the dataset')
    parser.add_argument('--directory', type=str, default='../data', help='Directory of the dataset')
    parser.add_argument('--hidden_size', type=int, default=128, help='Hidden size of the model')
    parser.add_argument('--max_length', type=int, default=15000, help='Maximum length of the sequences')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--batch_size', type=float, default=32, help='Batch size')
    parser.add_argument('--num_workers', type=float, default=4, help='Number of workers (should be 4*nb_GPU')
    parser.add_argument('--n_epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--print_every', type=int, default=10, help='Print every n epochs')
    parser.add_argument('--plot_every', type=int, default=10, help='Plot every n epochs')
    parser.add_argument('--save_every', type=int, default=10, help='Save every n epochs')
    parser.add_argument('--examples_every', type=int, default=5, help='Print examples every n epochs')
    parser.add_argument('--load_checkpoint', action='store_true', help='Load the best checkpoint if it exists')
    
    args = parser.parse_args()
    
    name = args.name
    hidden_size = args.hidden_size
    max_length = args.max_length
    lr = args.learning_rate
    weight_decay = args.weight_decay
    n_epochs = args.n_epochs
    print_every = args.print_every
    plot_every = args.plot_every
    save_every = args.save_every
    root_dir = args.directory
    batch_size = args.batch_size
    num_workers = args.num_workers
    load_checkpoint = args.load_checkpoint
    print_examples_every = args.print_examples_every
    
    optimizer_hyperparams = {
        'learning_rate': lr,
        'weight_decay': weight_decay,
        'n_epochs': n_epochs,
        'batch_size': batch_size,
        'num_workers': num_workers
    }
    
    model_hyperparams = {
        'hidden_size': hidden_size,
        'max_length': max_length
    }
    
    print_hyperparams = {
        'print_every': print_every,
        'plot_every': plot_every,
        'print_examples_every': print_examples_every
    }
    
    main(root_dir = root_dir,
        model_hyperparams=model_hyperparams,
        optimizer_hyperparams=optimizer_hyperparams,
        print_hyperparams=print_hyperparams,
        load_checkpoint=load_checkpoint,
        name=name
        )