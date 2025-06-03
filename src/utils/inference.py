import argparse
import os
import pickle
import re
import time
from pathlib import Path

import torch

from utils.models import AttnDecoderRNN, EncoderRNN, AttnDecoderRNN_packed, EncoderRNN_packed
from utils.processing import pad_sequences
from utils.training_utils import decode_data


def summarize_on_cpu(input_tensor, encoder, decoder, EOS_token, index2word, legacy):
    """
    Perform inference on a single input_tensor using encoder-decoder attention model on CPU.
    
    Args:
        input_tensor (torch.Tensor): Input tensor of shape [seq_len]
        encoder (EncoderRNN): Trained encoder model.
        decoder (AttnDecoderRNN): Trained decoder model.
        max_length (int): Maximum length of the output summary.
        SOS_token (int): Start-of-sequence token index.
        EOS_token (int): End-of-sequence token index.
        index2word (dict): Mapping from index to word.
        legacy (bool): Whether unpacked or packed sequences are being used
        
    Returns:
        str: Decoded summary sentence.
    """
    start_time = time.time()
    # Move models to CPU and set to eval mode
    device = torch.device("cpu")
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    encoder.eval()
    decoder.eval()
    
    input_tensor = torch.tensor(input_tensor, dtype=torch.long).unsqueeze(0).to(device)  # Add batch dimension
    target_tensor = None
    with torch.no_grad():
        if legacy:
            encoder_outputs, encoder_hidden = encoder(input_tensor)
            decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_tensor)
        else:
            input_lengths = torch.tensor(input_tensor.shape[1], dtype=torch.int64).unsqueeze(0)
            encoder_mask = (input_tensor != 0)
            encoder_outputs, encoder_hidden = encoder(input_tensor, input_lengths,encoder_mask)
            decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden,encoder_mask, target_tensor)


        # Get the predicted words
        _, topi = decoder_outputs.topk(1)
        decoded_words = decode_data(topi.squeeze(), index2word, EOS_token)
        
    # Split the decoded words into a list
    decoded_words = decoded_words.split()
    
    # Remove SOS and EOS tokens from the decoded words
    decoded_words = [word for word in decoded_words if word != 'SOS' and word != 'EOS']
    
    end_time = time.time()
    print(f"Inference completed in {end_time - start_time:.2f} seconds.")
    return ' '.join(decoded_words)

def main(root_dir, name, checkpoint_name, hidden_size, max_length, input_tensor, legacy):
    if legacy:
        dataset_dir = os.path.join(root_dir, 'data', name)
        save_dir = os.path.join(root_dir, 'checkpoints', name)
    else:
        dataset_dir = os.path.join(root_dir, 'data_packed', name)
        save_dir = os.path.join(root_dir, 'checkpoints_packed', name)

    checkpoint_path = os.path.join(save_dir, checkpoint_name)
    
    with open(os.path.join(dataset_dir, 'feature_tokenizer.pickle'), 'rb') as handle:
        feature_tokenizer = pickle.load(handle)
        
    num_words_text = max(feature_tokenizer.word2index.values()) + 1
    
    if legacy:
        encoder = EncoderRNN(num_words_text, hidden_size).to(torch.device('cpu'))
        decoder = AttnDecoderRNN(hidden_size, num_words_text, max_length).to(torch.device('cpu'))
    else:
        encoder = EncoderRNN_packed(num_words_text, hidden_size).to(torch.device('cpu'))
        decoder = AttnDecoderRNN_packed(hidden_size, num_words_text, max_length).to(torch.device('cpu'))

    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    encoder.load_state_dict(checkpoint['en'])
    decoder.load_state_dict(checkpoint['de'])
    
    # Preprocess the input tensor
    input_tensor = re.sub(r'[^a-zA-Z0-9\s]', '', input_tensor)
    input_tensor = input_tensor.lower()
    input_tensor = input_tensor.strip()
    
    # Tokenize the input tensor
    input_tensor = feature_tokenizer.texts_to_sequences([input_tensor])
    input_tensor = pad_sequences(input_tensor, maxlen=130)
    input_tensor = input_tensor[0]
    print("input_tensor idx:", input_tensor)
    summary = summarize_on_cpu(input_tensor, encoder, decoder,
                            EOS_token=feature_tokenizer.word2index['EOS'],
                            index2word=feature_tokenizer.index2word,
                            legacy=legacy)

    print("Summary:", summary)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Summarize text using a trained model.')
    parser.add_argument('--root_dir', type=str, default=Path.cwd().parent, help='Root directory of the project')
    parser.add_argument('--name', type=str, default='WikiHow', help='Name of the dataset')
    parser.add_argument('--legacy', type=bool, default=False, help="Whether to use packed sequences (legacy=False) or not (legacy=True)")
    parser.add_argument('--checkpoint_name', type=str, default='best_checkpoint.tar', help='Checkpoint file name')
    parser.add_argument('--hidden_size', type=int, default=128, help='Hidden size of the model')
    parser.add_argument('--max_length', type=int, default=50, help='Maximum length of the output summary')

    args = parser.parse_args()
    
    root_dir = args.root_dir
    name = args.name
    checkpoint_name = args.checkpoint_name
    hidden_size = args.hidden_size
    max_length = args.max_length
    legacy = args.legacy

    while True:
        # Ask for input tensor from the user
        input_tensor = input("Enter the text to summarize (or type 'exit' to quit): ")
        if input_tensor.lower() == 'exit':
            break
        # Don't forget to water your plants, they need it to survive.
        main(root_dir, name, checkpoint_name, hidden_size, max_length, input_tensor, legacy)