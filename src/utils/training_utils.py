import os
import random

import numpy as np
import torch
from ignite.metrics import Rouge
from matplotlib import pyplot as plt
from torcheval.metrics.functional import bleu_score


def plot_metrics(figures_dir, train_losses, val_losses, val_metrics, log_every, iterations_per_epoch):
    """
    Plots the training/validation losses and metrics per iteration.
    Marks the best validation point and each epoch boundary.
    """
    plt.rcParams.update({'font.size': 22})
    _, ax = plt.subplots(1, 2, figsize=(22, 9))

    nb_ticks = 50
    x_ticks = np.arange(1, len(train_losses) + 1, nb_ticks)
    best_val_idx = int(np.argmin(val_losses)) + 1

    # Plot training and validation losses
    ax[0].plot(np.arange(1, len(train_losses) + 1), train_losses, label='Training Loss')
    ax[0].plot(np.arange(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    ax[0].set_xticks(x_ticks)

    # Mark best validation loss
    ax[0].axhline(y=val_losses[best_val_idx - 1], color='red', linestyle='--', label='Best Val Loss')

    # Mark epoch lines
    max_iter = len(train_losses)
    for epoch in range(1, (max_iter * 1) // iterations_per_epoch + 1):
        x_pos = epoch * iterations_per_epoch // log_every
        ax[0].axvline(x=x_pos, color='gray', linestyle='--', linewidth=1)

    ax[0].set_xlabel('Iterations')
    ax[0].set_ylabel('Loss')
    ax[0].set_title('Losses per Iteration')
    ax[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)

    # Plot validation metrics
    for metric_name, metric_values in val_metrics.items():
        ax[1].plot(np.arange(1, len(metric_values) + 1), metric_values, label=metric_name)

    # Add epoch lines to metrics plot
    for epoch in range(1, (max_iter * 1) // iterations_per_epoch + 1):
        x_pos = epoch * iterations_per_epoch // log_every
        ax[1].axvline(x=x_pos, color='gray', linestyle='--', linewidth=1)

    ax[1].set_xlabel('Iterations')
    ax[1].set_ylabel('Metric Value')
    ax[1].set_title('Validation Metrics per Iteration')
    ax[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'metrics.png'))
    plt.show()

def evaluate_loss(dataloader, encoder, decoder, criterion):
    total_loss = 0
    total_batches = len(dataloader)

    with torch.no_grad():
        for idx, data in enumerate(dataloader):
            input_tensor, target_tensor = data

            encoder_outputs, encoder_hidden = encoder(input_tensor)
            decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_tensor)

            loss = criterion(
                decoder_outputs.view(-1, decoder_outputs.size(-1)),
                target_tensor.view(-1)
            )
            total_loss += loss.item()

            percent_complete = (idx / total_batches) * 100
            print("\r", end="")  # Move cursor up and clear line
            print(f'Evaluating loss: {percent_complete:.2f}%', end="")

    return total_loss / total_batches

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
    
    # Remove SOS and EOS tokens from predictions and targets
    predictions = [pred.replace('SOS', '').replace('EOS', '').strip() for pred in predictions]
    targets = [target.replace('SOS', '').replace('EOS', '').strip() for target in targets]
    
    try:
        metrics["BLEU"] = bleu_score(predictions, targets, n_gram=n1)
    except ValueError as e:
        print(f"Warning: BLEU score calculation skipped due to: {e}")
        metrics["BLEU"] = 0.0

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
    encoder.eval()
    decoder.eval()

    predictions = []
    targets = []
    total_batches = len(dataloader)

    with torch.no_grad():
        for idx, data in enumerate(dataloader):
            input_tensor, target_tensor = data

            predicted_words = make_predictions(encoder, decoder, input_tensor, index2word, EOS_token)
            target_words = decode_data(target_tensor[0], index2word, EOS_token)

            predictions.append(predicted_words)
            targets.append(target_words)

            percent_complete = (idx / total_batches) * 100
            print("\r", end="")  # Move cursor up and clear line
            print(f'Evaluating model: {percent_complete:.2f}%', end="")

    return compute_metrics(predictions, targets, n1=1, n2=2)

def inference_testing(encoder, decoder, dataloader, index2word, EOS_token, nb_decoding_test=5, writer=None):
    encoder.eval()
    decoder.eval()
    count_test = 0
    random_list = random.sample(range(len(dataloader)), nb_decoding_test)
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            if i in random_list:
                input_tensor, target_tensor = data
                decoded_words = make_predictions(encoder, decoder, input_tensor, index2word, EOS_token)
                input_text = decode_data(input_tensor[0], index2word, EOS_token)
                target_text = decode_data(target_tensor[0], index2word, EOS_token)

                print('Input: {}'.format(input_text))
                print('Target: {}'.format(target_text))
                print('Predicted: {}'.format(decoded_words))
                print('-----------------------------------')

                if writer:
                    writer.add_text(f'Examples/Input_{count_test}', input_text, i)
                    writer.add_text(f'Examples/Target_{count_test}', target_text, i)
                    writer.add_text(f'Examples/Predicted_{count_test}', decoded_words, i)

                count_test += 1
            if count_test == nb_decoding_test:
                break
            
def print_metrics(metrics, writer=None):
    """
    Prints the metrics and optionally logs them to TensorBoard.
    """
    print('-----------------------------------')
    for key in ["BLEU", "Rouge-L-F", "Rouge-1-F", "Rouge-2-F"]:
        if writer:
            writer.add_scalar(f'Metrics/{key}', metrics[key])
        print('{}: {:.4f}'.format(f"{key} score", metrics[key]))
    print('-----------------------------------')

