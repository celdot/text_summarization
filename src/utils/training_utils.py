import os
import random

import torch
from ignite.metrics import Rouge
from matplotlib import pyplot as plt
from torcheval.metrics.functional import bleu_score


def plot_metrics(figures_dir, train_losses, val_losses, val_metrics):
    """
    Plots the training and validation losses.
    """
    plt.rcParams.update({'font.size': 22})
    fig, ax = plt.subplots(1, 2, figsize=(16, 9))
    # Plot training and validation losses
    ax[0].plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
    ax[0].plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss')
    ax[0].set_title('Training and Validation Losses')
    ax[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)

    # Plot validation metrics
    for metric_name, metric_values in val_metrics.items():
        ax[1].plot(range(1, len(metric_values) + 1), metric_values, label=metric_name)
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Metric Value')
    ax[1].set_title('Validation Metrics')
    # Put the legend below the plot, centered, with 2 columns
    ax[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'metrics.png'))
    # plt.close(fig)
    plt.show()

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

            predicted_words = make_predictions(encoder, decoder, input_tensor, index2word, EOS_token)
            target_words = decode_data(target_tensor[0], index2word, EOS_token)

            predictions.append(predicted_words)
            targets.append(target_words)

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

