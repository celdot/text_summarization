import random

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from utils.models import AttnDecoderRNN, EncoderRNN
from utils.training_utils import (evaluate_loss, evaluate_model,
                                  inference_testing, plot_metrics,
                                  print_metrics)

plt.switch_backend('agg')
import argparse
import copy
import os
import pickle
from itertools import product
from pathlib import Path

import numpy as np
import optuna
from optuna.visualization import (plot_optimization_history,
                                  plot_param_importances, plot_slice)
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


def training_loop(train_dataloader, val_dataloader, encoder, decoder, criterion,
          index2words, EOS_token, checkpoint_path, figures_dir, optimizer_hyperparams,
          saved_metrics, print_examples_every, tuning):
    
    learning_rate = optimizer_hyperparams['learning_rate']
    weight_decay = optimizer_hyperparams['weight_decay']
    n_epochs = optimizer_hyperparams['n_epochs']
    early_stopping_patience = optimizer_hyperparams["early_stopping_patience"]
    
    # Load saved metrics if available
    start_epoch = saved_metrics.get('start_epoch')
    plot_train_losses = saved_metrics.get('train_losses')
    plot_val_losses = saved_metrics.get('val_losses')
    best_val_loss = saved_metrics.get('best_val_loss')
    plot_val_metrics = saved_metrics.get('val_metrics')

    # Initialize TensorBoard
    if not tuning:
        writer = SummaryWriter(log_dir='tensorboard_logs')

    # Initializations
    print('Initializing ...')
    no_improvement_count = 0

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate, weight_decay=weight_decay)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate, weight_decay=weight_decay)

    print("Training...")
    for epoch in tqdm(range(start_epoch, n_epochs + 1)):
        training_loss = train_epoch(train_dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)

        val_loss = evaluate_loss(val_dataloader, encoder, decoder, criterion)

        if val_loss < best_val_loss:
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)
            best_val_loss = val_loss
            no_improvement_count = 0
            torch.save({
                'epoch': epoch,
                'en': encoder.state_dict(),
                'de': decoder.state_dict(),
                'train_losses': plot_train_losses,
                'val_losses': plot_val_losses,
                'best_val_loss': best_val_loss,
                'val_metrics': plot_val_metrics
            }, checkpoint_path)
        else:
            no_improvement_count += 1
            if no_improvement_count >= early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break

        if not tuning:
            writer.add_scalar('Loss/Train', training_loss, epoch)
            writer.add_scalar('Loss/Validation', val_loss, epoch)
            print('Epoch: {}; Avg train loss: {:.4f}; Avg val loss: {:.4f}.'.format(
                epoch, training_loss, val_loss))
            
            val_metrics = evaluate_model(encoder, decoder, val_dataloader, index2words, EOS_token)
            for metric_name in plot_val_metrics.keys():
                plot_val_metrics[metric_name].append(val_metrics[metric_name])
            print_metrics(val_metrics, writer)

            if epoch % print_examples_every == 0:
                inference_testing(encoder, decoder, val_dataloader, index2words, EOS_token, nb_decoding_test=5, writer=writer)

            plot_train_losses.append(training_loss)
            plot_val_losses.append(val_loss)

    if not tuning:
        writer.close()
        plot_metrics(figures_dir, plot_train_losses, plot_val_losses, plot_val_metrics)
        
def train(root_dir, checkpoint_path, feature_tokenizer, device, name, model_hyperparams,
                  optimizer_hyperparams, print_examples_every, tuning,
                  load_checkpoint=False):
    """
    Main training loop for the Seq2Seq model with attention.
    """
    # Get directories
    dataset_dir = os.path.join(root_dir, 'data', name)
    figures_dir = os.path.join(root_dir, 'figures', name)

    os.makedirs(figures_dir, exist_ok=True)
    
    # Get the hyperparameters
    batch_size = optimizer_hyperparams['batch_size']
    num_workers = optimizer_hyperparams['num_workers']
    max_length = model_hyperparams['max_length']
    hidden_size = model_hyperparams['hidden_size']

    # Load the dataset
    X_train = torch.load(os.path.join(dataset_dir, "x_train.pt"))
    X_val = torch.load(os.path.join(dataset_dir, "x_val.pt"))
    y_train = torch.load(os.path.join(dataset_dir, "y_train.pt"))
    y_val = torch.load(os.path.join(dataset_dir, "y_val.pt"))

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

    index2words = feature_tokenizer.index2word
    num_words_text = max(feature_tokenizer.word2index.values()) + 1
    EOS_token = feature_tokenizer.word2index.get("EOS", 2)

    # Initialize the model
    print("Initialize the model")
    encoder = EncoderRNN(num_words_text, hidden_size).to(device)
    decoder = AttnDecoderRNN(hidden_size, num_words_text, max_length).to(device)
    criterion = nn.NLLLoss(ignore_index=0)

    start_epoch = 1
    plot_train_losses = []
    plot_val_losses = []
    best_val_loss = float('inf')
    plot_val_metrics = {"BLEU": [], "Rouge-L-F": [], "Rouge-1-F": [], "Rouge-2-F": []}

    if load_checkpoint:
        print("Loading checkpoint...")
        checkpoint = torch.load(checkpoint_path)
        encoder.load_state_dict(checkpoint['en'])
        decoder.load_state_dict(checkpoint['de'])
        start_epoch = checkpoint['epoch'] + 1  # Resume from next epoch
        plot_train_losses = checkpoint.get('train_losses', [])
        plot_val_losses = checkpoint.get('val_losses', [])
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        plot_val_metrics = checkpoint.get('val_metrics', {"BLEU": [], "Rouge-L-F": [], "Rouge-1-F": [], "Rouge-2-F": []})
        
    saved_metrics = {
        'start_epoch': start_epoch,
        'train_losses': plot_train_losses,
        'val_losses': plot_val_losses,
        'best_val_loss': best_val_loss,
        'val_metrics': plot_val_metrics
    }

    # Train the model
    training_loop(train_dataloader, val_dataloader, encoder, decoder, criterion,
      index2words, EOS_token, checkpoint_path, figures_dir,
      optimizer_hyperparams, saved_metrics, print_examples_every, tuning)
        
def evaluate(root_dir, name, device, feature_tokenizer, checkpoint_path, batch_size, model_hyperparams):
    """
    Evaluate the model on the test set.
    """
    dataset_dir = os.path.join(root_dir, 'data', name)
    
    X_test = torch.load(os.path.join(dataset_dir, "x_test.pt"))
    y_test = torch.load(os.path.join(dataset_dir, "y_test.pt"))
    
    test_dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_test, y_test),
        batch_size=batch_size,
        shuffle=True,
    )

    num_words_text = max(feature_tokenizer.word2index.values()) + 1
    EOS_token = feature_tokenizer.word2index.get("EOS", 2)
    
    hidden_size = model_hyperparams['hidden_size']
    max_length = model_hyperparams['max_length']
    
    encoder = EncoderRNN(num_words_text, hidden_size).to(device)
    decoder = AttnDecoderRNN(hidden_size, num_words_text, max_length).to(device)
    criterion = nn.NLLLoss(ignore_index=0)

    # Load the best model
    checkpoint = torch.load(checkpoint_path)
    encoder.load_state_dict(checkpoint['en'])
    decoder.load_state_dict(checkpoint['de'])

    # Test the model
    print('Evaluating the model on the test set...')
    test_loss = evaluate_loss(test_dataloader, encoder, decoder, criterion)
    print('Test loss: {:.4f}'.format(test_loss))

    # Evaluate the model
    metrics = evaluate_model(encoder, decoder, test_dataloader, feature_tokenizer.index2word, EOS_token)
    print_metrics(metrics)
    # Get a random sample from the test set
    inference_testing(encoder, decoder, test_dataloader, feature_tokenizer.index2word, EOS_token, nb_decoding_test=5)

def main(root_dir,
    model_hyperparams,
    optimizer_hyperparams,
    print_examples_every,
    tuning,
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
    
    dataset_dir = os.path.join(root_dir, 'data', name)
    save_dir = os.path.join(root_dir, 'checkpoints', name)
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, 'best_checkpoint.tar')
    
    # Load the vocabulary
    print("Loading tokenizer")
    with open(os.path.join(dataset_dir, 'feature_tokenizer.pickle'), 'rb') as handle:
            feature_tokenizer = pickle.load(handle)
            
    train(
        root_dir=root_dir, checkpoint_path=checkpoint_path,
        feature_tokenizer=feature_tokenizer, device=device, name=name,
        model_hyperparams=model_hyperparams,
        optimizer_hyperparams=optimizer_hyperparams,
        print_examples_every=print_examples_every,
        tuning=tuning, load_checkpoint=load_checkpoint)

    evaluate(root_dir=root_dir, name=name, device=device, feature_tokenizer=feature_tokenizer,
             checkpoint_path=checkpoint_path, batch_size=optimizer_hyperparams['batch_size'], model_hyperparams=model_hyperparams)

def objective(root_dir, name, trial):
    # Define hyperparameter search space
    hidden_size = trial.suggest_categorical('hidden_size', [64, 128, 256])
    max_length = trial.suggest_categorical('max_length', [30, 50, 100])
    learning_rate = trial.suggest_categorical('learning_rate', [1e-3, 1e-4, 1e-5])
    weight_decay = trial.suggest_categorical('weight_decay', [1e-4, 1e-5, 1e-6])
    early_stopping_patience = trial.suggest_categorical('early_stopping_patience', [3, 5, 10])

    # Wrap parameters
    optimizer_hyperparams = {
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'n_epochs': 3,
        'batch_size': 32,
        'num_workers': 4,
        'early_stopping_patience': early_stopping_patience
    }

    model_hyperparams = {
        'hidden_size': hidden_size,
        'max_length': max_length
    }

    # Set a temp directory to avoid overwriting
    trial_name = f"trial_{trial.number}"
    trial_dir = os.path.join(root_dir, 'parameters_tuning', name)

    os.makedirs(trial_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Random seed for reproducibility
    random.seed(5719)
    np.random.seed(5719)
    torch.manual_seed(5719)
    torch.use_deterministic_algorithms(False)
    
    dataset_dir = os.path.join(root_dir, 'data', name)
    
    # Load the vocabulary
    with open(os.path.join(dataset_dir, 'feature_tokenizer.pickle'), 'rb') as handle:
            feature_tokenizer = pickle.load(handle)
    
    # Run training (validation loss is used as the objective)
    try:
        checkpoint_path = os.path.join(trial_dir, "checkpoints", trial_name, "best_checkpoint.tar")
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        train(
        root_dir=root_dir, checkpoint_path=checkpoint_path,
        feature_tokenizer=feature_tokenizer, device=device, name=name,
        model_hyperparams=model_hyperparams,
        optimizer_hyperparams=optimizer_hyperparams,
        print_examples_every=1000,
        tuning=True, load_checkpoint=False)

        # Load best checkpoint to get best val loss
        checkpoint = torch.load(checkpoint_path)
        return checkpoint.get('best_val_loss', float('inf'))

    except Exception as e:
        print(f"Trial {trial.number} failed with error: {e}")
        return float('inf')

def tuning(root_dir, nb_trials, name):
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(root_dir, name, trial), n_trials=nb_trials)

    print("Best trial:")
    trial = study.best_trial
    print(f"Value (Validation Loss): {trial.value}")
    print("Params: ")
    for key, value in trial.params.items():
        print(f"{key}: {value}")
        
    # Save the study results
    study_dir = os.path.join(root_dir, 'parameters_tuning', name, 'study_results')
    os.makedirs(study_dir, exist_ok=True)

    # Save the optimization history
    fig = plot_optimization_history(study)
    fig.savefig(os.path.join(study_dir, 'optimization_history.png'))
    plt.close(fig)

    # Save the parameter importances
    fig = plot_param_importances(study)
    fig.savefig(os.path.join(study_dir, 'param_importances.png'))
    plt.close(fig)

    # Save the slice plot
    fig = plot_slice(study)
    fig.savefig(os.path.join(study_dir, 'slice_plot.png'))
    plt.close(fig)
            
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
    parser.add_argument('--print_example_every', type=int, default=5, help='Print examples every n epochs')
    parser.add_argument('--early_stopping_patience', type=int, default=5, help='Early stopping patience')
    parser.add_argument('--hyp_tuning', action='store_true', help='Run hyperparameter tuning with Optuna')
    parser.add_argument('--num_trials', type=int, default=20, help='Number of trials for hyperparameter tuning')
    
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
    print_example_every = args.print_example_every
    early_stopping_patience = args.early_stopping_patience
    hyp_tuning = args.hyp_tuning
    num_trials = args.num_trials
        
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
    
    root_dir = Path.cwd().parent
    
    main(root_dir = root_dir,
        model_hyperparams=model_hyperparams,
        tuning = hyp_tuning,
        optimizer_hyperparams=optimizer_hyperparams,
        print_examples_every=print_example_every,
        load_checkpoint=load_checkpoint,
        name=name
        )
    
    if hyp_tuning:
        tuning(root_dir=root_dir, nb_trials=num_trials, name=name)

