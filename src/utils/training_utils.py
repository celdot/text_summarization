import os
import random

import numpy as np
import torch
from ignite.metrics import Rouge
from matplotlib import pyplot as plt
from tqdm import tqdm
from torcheval.metrics.functional import bleu_score

def train_epoch_packed(
    dataloader, encoder, decoder, encoder_optimizer,
    decoder_optimizer, criterion, val_dataloader, iteration_counter,
    log_every, print_examples_every, index2words, EOS_token,
    plot_train_losses, plot_val_losses, plot_val_metrics,
    tuning, checkpoint_path, best_val_loss_ref, no_improvement_count,
    early_stopping_patience
):
    encoder.train()
    decoder.train()

    total_loss = 0
    
    early_stop = False

    for batch_idx, data in enumerate(tqdm(dataloader)):
        if batch_idx < iteration_counter % len(dataloader):
            continue
        # NOTE: THESE ARE NOT PACKED YET!!
        input_tensor, input_lengths, target_tensor, target_lengths = data

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        encoder_mask = (input_tensor != 0)
        encoder_outputs, encoder_hidden = encoder(input_tensor, input_lengths, encoder_mask)
        decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, encoder_mask, target_tensor)

        loss = criterion(
            decoder_outputs.view(-1, decoder_outputs.size(-1)),
            target_tensor.view(-1)
        )
        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        total_loss += loss.item()

        if iteration_counter % log_every == 0:
            avg_train_loss = total_loss / (batch_idx + 1)
            val_loss = evaluate_loss(val_dataloader, encoder, decoder, criterion, legacy=False)

            plot_train_losses.append(avg_train_loss)
            plot_val_losses.append(val_loss)

            # Early stopping & checkpointing
            if val_loss < best_val_loss_ref:
                if os.path.exists(checkpoint_path):
                    os.remove(checkpoint_path)
                best_val_loss_ref = val_loss
                no_improvement_count = 0
                torch.save({
                    'iteration': iteration_counter,
                    'en': encoder.state_dict(),
                    'de': decoder.state_dict(),
                    'train_losses': plot_train_losses,
                    'val_losses': plot_val_losses,
                    'best_val_loss': best_val_loss_ref,
                    'val_metrics': plot_val_metrics
                }, checkpoint_path)
            else:
                no_improvement_count += 1
                if no_improvement_count >= early_stopping_patience:
                    print(f"Early stopping triggered at iteration {iteration_counter}")
                    early_stop = True  # signal to stop training

            if not tuning:
                print(f"[Iter {iteration_counter}] Train loss: {avg_train_loss:.4f}, Val loss: {val_loss:.4f}")

            encoder.train()
            decoder.train()
            
        if iteration_counter % print_examples_every == 0 and not tuning:
            val_metrics = evaluate_model(encoder, decoder, val_dataloader, index2words, EOS_token, legacy=False)
            for metric_name in plot_val_metrics.keys():
                plot_val_metrics[metric_name].append(val_metrics[metric_name])
                print_metrics(val_metrics)
                
            inference_testing(encoder, decoder, val_dataloader, index2words, EOS_token, nb_decoding_test=5, legacy=False)
                
            encoder.train()
            decoder.train()

        iteration_counter += 1

    return early_stop


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

def evaluate_loss(dataloader, encoder, decoder, criterion, legacy=False):
    total_loss = 0
    total_batches = len(dataloader)

    with torch.no_grad():
        if legacy:
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
        else:
            for idx, data in enumerate(dataloader):
                input_tensor, input_lengths, target_tensor, target_lengths = data

                encoder_mask = (input_tensor != 0)

                encoder_outputs, encoder_hidden = encoder(input_tensor, input_lengths, encoder_mask)
                decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, encoder_mask, target_tensor)

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

def make_predictions(encoder, decoder, input_tensor, index2word, EOS_token, legacy, input_lengths):
    """
    Computes the summary for the given input tensor.
    NOTE: If legacy==False => must provide input_lengths!!
    """
    input_tensor = input_tensor[0].unsqueeze(0)
    target_tensor = None  # Set target_tensor to None for inference

    
    if legacy:
        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_tensor)
    else:
        encoder_mask = (input_tensor != 0)
        # Shorten input_lengths as well
        input_lengths = input_lengths[0].unsqueeze(0)
        encoder_outputs, encoder_hidden = encoder(input_tensor, input_lengths, encoder_mask)
        decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, encoder_mask, target_tensor)


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

def evaluate_model(encoder, decoder, dataloader, index2word, EOS_token, legacy=False):
    encoder.eval()
    decoder.eval()

    predictions = []
    targets = []
    total_batches = len(dataloader)

    with torch.no_grad():
        if legacy:
            for idx, data in enumerate(dataloader):
                input_tensor, target_tensor = data

                predicted_words = make_predictions(encoder, decoder, input_tensor, index2word, EOS_token)
                target_words = decode_data(target_tensor[0], index2word, EOS_token)

                predictions.append(predicted_words)
                targets.append(target_words)

                percent_complete = (idx / total_batches) * 100
                print("\r", end="")  # Move cursor up and clear line
                print(f'Evaluating model: {percent_complete:.2f}%', end="")
        else:
            for idx, data in enumerate(dataloader):
                input_tensor, input_lengths, target_tensor, target_lengths = data
            

                predicted_words = make_predictions(encoder, decoder, input_tensor, index2word, EOS_token, legacy=legacy, input_lengths=input_lengths)
                target_words = decode_data(target_tensor[0], index2word, EOS_token)

                predictions.append(predicted_words)
                targets.append(target_words)

                percent_complete = (idx / total_batches) * 100
                print("\r", end="")  # Move cursor up and clear line
                print(f'Evaluating model: {percent_complete:.2f}%', end="")

    return compute_metrics(predictions, targets, n1=1, n2=2)

def inference_testing(encoder, decoder, dataloader, index2word, EOS_token, nb_decoding_test=5, writer=None, legacy=False):
    encoder.eval()
    decoder.eval()
    count_test = 0
    random_list = random.sample(range(len(dataloader)), nb_decoding_test)
    with torch.no_grad():
        if legacy:
            for i, data in enumerate(dataloader):
                if i in random_list:
                    input_tensor, target_tensor = data
                    decoded_words = make_predictions(encoder, decoder, input_tensor, index2word, EOS_token, legacy=legacy)
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
        else:
            for i, data in enumerate(dataloader):
                if i in random_list:
                    input_tensor, input_lengths, target_tensor, target_lengths = data
                    decoded_words = make_predictions(encoder, decoder, input_tensor, index2word, EOS_token, legacy=legacy, input_lengths=input_lengths)
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

