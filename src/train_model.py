import random

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from utils.models import AttnDecoderRNN, EncoderRNN
from utils.training_utils import (evaluate_loss, evaluate_model,
                                  inference_testing, plot_metrics)

plt.switch_backend('agg')
import argparse
import copy
import os
import pickle
from itertools import product

import numpy as np
from torch.utils.tensorboard import SummaryWriter


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
          print_examples_every):
    
    learning_rate = optimizer_hyperparams['learning_rate']
    weight_decay = optimizer_hyperparams['weight_decay']
    n_epochs = optimizer_hyperparams['n_epochs']
    early_stopping_patience = optimizer_hyperparams["early_stopping_patience"]

    # Initialize TensorBoard
    writer = SummaryWriter(log_dir=os.path.join(figures_dir, 'tensorboard_logs'))

    # Initializations
    print('Initializing ...')
    plot_train_losses = []
    plot_val_losses = []
    plot_val_metrics = {"BLEU": [], "Rouge-L-F": [], "Rouge-1-F": [], "Rouge-2-F": []}
    print_train_loss_total = 0
    plot_train_loss_total = 0
    print_val_loss_total = 0
    plot_val_loss_total = 0
    best_val_loss = float('inf')
    no_improvement_count = 0

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate, weight_decay=weight_decay)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate, weight_decay=weight_decay)

    print("Training...")
    for epoch in range(1, n_epochs + 1):
        training_loss = train_epoch(train_dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_train_loss_total += training_loss
        plot_train_loss_total += training_loss

        val_loss = evaluate_loss(val_dataloader, encoder, decoder, criterion)
        print_val_loss_total += val_loss
        plot_val_loss_total += val_loss

        writer.add_scalar('Loss/Train', training_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improvement_count = 0
            torch.save({
                'epoch': epoch,
                'en': encoder.state_dict(),
                'de': decoder.state_dict(),
            }, os.path.join(save_directory, 'best_checkpoint.tar'))
        else:
            no_improvement_count += 1
            if no_improvement_count >= early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break

        print('Epoch: {}; Avg train loss: {:.4f}; Avg val loss: {:.4f}.'.format(
                epoch, print_train_loss_total, print_val_loss_total))

        val_metrics = evaluate_model(encoder, decoder, val_dataloader, index2words, EOS_token)
        print('-----------------------------------')
        for key in plot_val_metrics.keys():
            plot_val_metrics[key].append(val_metrics[key])
            writer.add_scalar(f'Metric/{key}', val_metrics[key], epoch)
            print('{}: {:.4f}'.format(f"{key} score", val_metrics[key]))
        print('-----------------------------------')

        print_train_loss_total = 0
        print_val_loss_total = 0

        if epoch % print_examples_every == 0:
            inference_testing(encoder, decoder, val_dataloader, index2words, EOS_token, nb_decoding_test=5, writer=writer)

        plot_train_losses.append(plot_train_loss_total)
        plot_val_losses.append(plot_val_loss_total)
        plot_train_loss_total = 0
        plot_val_loss_total = 0

    writer.close()
    plot_metrics(figures_dir, plot_train_losses, plot_val_losses, plot_val_metrics)

def main(root_dir,
    model_hyperparams,
    optimizer_hyperparams,
    print_examples_every,
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
          optimizer_hyperparams, print_examples_every)

    # Load the best model
    checkpoint = torch.load(os.path.join(save_dir, 'best_checkpoint.tar'))
    encoder.load_state_dict(checkpoint['en'])
    decoder.load_state_dict(checkpoint['de'])

    # Test the model
    print('Evaluating the model on the test set...')
    test_loss = evaluate_loss(test_dataloader, encoder, decoder, criterion)
    print('Test loss: {:.4f}'.format(test_loss))

    # Evaluate the model
    metrics = evaluate_model(encoder, decoder, test_dataloader, feature_tokenizer.index2word, EOS_token)
    print('-----------------------------------')
    for key in ["BLEU", "Rouge-L-F", "Rouge-1-F", "Rouge-2-F"]:
        print('{}: {:.4f}'.format(f"{key} score", metrics[key]))
    print('-----------------------------------')
    # Get a random sample from the test set
    inference_testing(encoder, decoder, test_dataloader, feature_tokenizer.index2word, EOS_token, nb_decoding_test=5)
            
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
    parser.add_argument('--load_checkpoint', action='store_true', help='Load the best checkpoint if it exists')
    parser.add_argument('--print_examples_every', type=int, default=5, help='Print examples every n epochs')
    parser.add_argument('--early_stopping_patience', type=int, default=5, help='Early stopping patience')
    
    args = parser.parse_args()
    
    name = args.name
    hidden_size = args.hidden_size
    max_length = args.max_length
    lr = args.learning_rate
    weight_decay = args.weight_decay
    n_epochs = args.n_epochs
    root_dir = args.directory
    batch_size = args.batch_size
    num_workers = args.num_workers
    load_checkpoint = args.load_checkpoint
    print_examples_every = args.print_examples_every
    early_stopping_patience = args.early_stopping_patience
    
    optimizer_hyperparams = {
        'learning_rate': lr,
        'weight_decay': weight_decay,
        'n_epochs': n_epochs,
        'batch_size': batch_size,
        'num_workers': num_workers,
        'early_stopping_patience': early_stopping_patience
    }
    
    model_hyperparams = {
        'hidden_size': hidden_size,
        'max_length': max_length
    }
    
    main(root_dir = root_dir,
        model_hyperparams=model_hyperparams,
        optimizer_hyperparams=optimizer_hyperparams,
        print_examples_every=print_examples_every,
        load_checkpoint=load_checkpoint,
        name=name
        )
    
    hyp_tuning = True
    param_grid = {
        'hidden_size': [64, 128, 256],
        'max_length': [30, 50, 100],
        'learning_rate': [1e-3, 1e-4, 1e-5],
        'weight_decay': [1e-4, 1e-5, 1e-6],
        'n_epochs': [20, 50, 100],
        'early_stopping_patience': [3, 5, 10],
    }

            