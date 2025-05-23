import os
import random

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ignite.metrics import Rouge
from torcheval.metrics.functional.text import bleu
from tqdm import tqdm

from utils.models import AttnDecoderRNN, EncoderRNN
from utils.processing import processing_pipeline

plt.switch_backend('agg')
import argparse
import pickle
from pathlib import Path

import matplotlib.ticker as ticker
import numpy as np


def plot_losses(figures_dir, train_losses, val_losses):
    """
    Plots the training and validation losses.
    """
    fig = plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.savefig(os.path.join(figures_dir, 'losses.png'))
    plt.close(fig)
    
def evaluate_loss(dataloader, encoder, decoder, criterion):
    total_loss = 0
    with torch.no_grad():
        for data in dataloader:
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
    decoded_words = []
    for idx in text_ids:
        if idx.item() == EOS_token:
            decoded_words.append('<EOS>')
            break
        decoded_words.append(index2word[idx.item()])
    
    return decoded_words

def compute_summary(encoder, decoder, input_tensor, target_tensor, index2word, EOS_token):
    """
    Computes the summary for the given input tensor and target tensor.
    """
    input_tensor = input_tensor[0].unsqueeze(0)
    target_tensor = target_tensor[0].unsqueeze(0)

    encoder_outputs, encoder_hidden = encoder(input_tensor)
    decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_tensor)

    # Get the predicted words
    _, topi = decoder_outputs.topk(1)
    decoded_words = decode_data(topi.squeeze(), index2word, EOS_token)

    # Get the target words
    target_words = decode_data(target_tensor, index2word, EOS_token)

    return decoded_words, target_words

def compute_metrics(predictions, targets, n):
    """
    Computes the BLEU score and ROUGE score for the predictions and targets.
    """
    rouge_metrics = Rouge(variants=["L", "S", n], multiref="best")

    rouge_metrics.update(([predictions], [targets]))
    metrics = rouge_metrics.compute()
    metrics["bleu"] = bleu_score(predictions, targets, max_n=n, weights=[1/n]*n)

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
        for data in dataloader:
            input_tensor, target_tensor = data

            decoded_words, target_words = compute_summary(encoder, decoder, input_tensor, target_tensor, index2word, EOS_token)

            predictions.append(decoded_words)
            targets.append(target_words)

    return compute_metrics(predictions, targets, n=2)

def train_epoch(dataloader, encoder, decoder, encoder_optimizer,
          decoder_optimizer, criterion):

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
          index2words, EOS_token, save_directory, figures_dir,
          n_epochs= 50, learning_rate=0.001, weight_decay=1e-5,
          print_every=100, plot_every=100, save_every=100, print_examples_every=5):

    # Initializations
    print('Initializing ...')
    plot_train_losses = []
    plot_val_losses = []
    print_train_loss_total = 0  # Reset every print_every
    plot_train_loss_total = 0  # Reset every plot_every
    print_val_loss_total = 0  # Reset every print_every
    plot_val_loss_total = 0  # Reset every plot_every

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

        # Print progress
        if epoch % print_every == 0:
                # Compute metrics for validation set
            val_metrics = evaluate_model(encoder, decoder, val_dataloader, index2words, EOS_token)
            print_train_loss_total = print_train_loss_total / print_every
            print_val_loss_total = print_val_loss_total / print_every
            print('epoch: {}; Average training loss: {:.4f}; Average validation loss: {:.4f}.'.format(
                    epoch, epoch / n_epochs * 100, print_train_loss_total, print_val_loss_total))
        if epoch % print_examples_every == 0:
            print('BLEU score: {:.4f}'.format(val_metrics['bleu']))
            print('ROUGE-L score: {:.4f}'.format(val_metrics['Rouge-L-F']))
            print('ROUGE-S score: {:.4f}'.format(val_metrics['Rouge-S-F']))
            print('ROUGE-2 score: {:.4f}'.format(val_metrics['Rouge-2-F']))
            print('-----------------------------------')
            print_train_loss_total = 0
            print_val_loss_total = 0
            # Get a random sample from the validation set
            index = random.randint(0, len(val_dataloader) - 1)
            for i, data in enumerate(val_dataloader):
                if i == index:
                    input_tensor, target_tensor = data
                    break
            decoded_words, target_words = compute_summary(encoder, decoder, input_tensor, target_tensor, index2words, EOS_token)
            print('Input: {}'.format(decode_data(input_tensor, index2words, EOS_token)))
            print('Target: {}'.format(target_words))
            print('Predicted: {}'.format(decoded_words))
            print('-----------------------------------')

        # Plot loss progress
        if epoch % plot_every == 0:
            plot_loss_avg = plot_train_loss_total / plot_every
            plot_train_losses.append(plot_loss_avg)
            plot_val_loss_avg = plot_val_loss_total / plot_every
            plot_val_losses.append(plot_val_loss_avg)
            plot_train_loss_total = 0
            plot_val_loss_total = 0
            
        # Save checkpoint
        if (epoch % save_every == 0):
            torch.save({
                'epoch': epoch,
                'en': encoder.state_dict(),
                'de': decoder.state_dict(),
            }, os.path.join(save_directory, '{}_checkpoint.tar'.format(epoch)))
            
        # Save the best model
        if (epoch == 1) or (val_loss < best_val_loss):
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'en': encoder.state_dict(),
                'de': decoder.state_dict(),
            }, os.path.join(save_directory, 'best_checkpoint.tar'))

    plot_losses(figures_dir, plot_train_losses, plot_val_losses)
    
def main(root_dir,
         EOS_token,
    hidden_size = 128,
    name = "WikiHow",
    max_length = 50,
    lr = 0.001,
    weight_decay = 1e-4,
    batch_size = 32,
    num_workers = 4,
    n_epochs = 50,
    print_every = 10,
    plot_every = 10,
    save_every = 10,
    print_examples_every=5,
    ):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    
    # Random seed for reproducibility
    random.seed(5719)
    np.random.seed(5719)
    torch.manual_seed(5719)
    torch.use_deterministic_algorithms(True)
    
    # Get directories
    dataset_dir = os.path.join(root_dir, 'data', name)
    save_dir = os.path.join(root_dir, 'checkpoints', name)
    figures_dir = os.path.join(root_dir, 'figures', name)
    
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    
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
    with open(os.path.join(dataset_dir, 'feature_tokenizer.pickle'), 'rb') as handle:
            feature_tokenizer = pickle.load(handle)
    with open(os.path.join(dataset_dir, 'label_tokenizer.pickle'), 'rb') as handle:
            label_tokenizer = pickle.load(handle)
            
    num_words_text = len(feature_tokenizer.word_index) + 1 # because we are using 1-based indexing (0 is reserved for padding)
    num_words_summary = len(label_tokenizer.word_index) + 1

    # Initialize the model
    encoder = EncoderRNN(num_words_text, hidden_size).to(device)
    decoder = AttnDecoderRNN(hidden_size, num_words_summary, max_length).to(device)
    criterion = nn.NLLLoss(ignore_index=0)

    # Train the model
    train(train_dataloader, val_dataloader, encoder, decoder, criterion,
          feature_tokenizer.index_word, EOS_token, save_dir, figures_dir,
          learning_rate=lr, weight_decay=weight_decay, n_epochs=n_epochs,
          print_every=print_every, plot_every=plot_every, save_every=save_every, print_examples_every=print_examples_every)
    
    # Load the best model
    checkpoint = torch.load(os.path.join(save_dir, 'best_checkpoint.tar'))
    encoder.load_state_dict(checkpoint['en'])
    decoder.load_state_dict(checkpoint['de'])
    
    # Test the model
    test_loss = evaluate_loss(test_dataloader, encoder, decoder, criterion)
    print('Test loss: {:.4f}'.format(test_loss))
    
    # Evaluate the model
    metrics = evaluate_model(encoder, decoder, test_dataloader, feature_tokenizer.index_word, EOS_token)
    print('BLEU score: {:.4f}'.format(metrics['bleu']))
    print('ROUGE-L score: {:.4f}'.format(metrics['Rouge-L-F']))
    print('ROUGE-S score: {:.4f}'.format(metrics['Rouge-S-F']))
    print('ROUGE-2 score: {:.4f}'.format(metrics['Rouge-2-F']))
    print('-----------------------------------')
    # Get a random sample from the test set
    index = random.randint(0, len(test_dataloader) - 1)
    for i, data in enumerate(test_dataloader):
        if i == index:
            input_tensor, target_tensor = data
            break
    decoded_words, target_words = compute_summary(encoder, decoder, input_tensor, target_tensor, feature_tokenizer.index_word, EOS_token)
    print('Input: {}'.format(decode_data(input_tensor, feature_tokenizer.index_word, EOS_token)))
    print('Target: {}'.format(decode_data(target_tensor, feature_tokenizer.index_word, EOS_token)))
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
    
    main(root_dir = root_dir,
         hidden_size=hidden_size,
         name=name,
         max_length=max_length,
         lr=lr,
         weight_decay=weight_decay,
         batch_size = batch_size,
         num_workers = num_workers, 
         n_epochs=n_epochs,
         print_every=print_every,
         plot_every=plot_every,
         save_every=save_every)