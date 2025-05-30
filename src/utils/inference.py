import os
import pickle
from pathlib import Path

import torch
from models import AttnDecoderRNN, EncoderRNN
from training_utils import decode_data


def summarize_on_cpu(input_tensor, encoder, decoder, EOS_token, index2word):
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
        
    Returns:
        str: Decoded summary sentence.
    """

    # Move models to CPU and set to eval mode
    device = torch.device("cpu")
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    encoder.eval()
    decoder.eval()

    input_tensor = input_tensor[0].unsqueeze(0).to(device)
    target_tensor = None
    with torch.no_grad():
        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_tensor)

        # Get the predicted words
        _, topi = decoder_outputs.topk(1)
        decoded_words = decode_data(topi.squeeze(), index2word, EOS_token)

    return ' '.join(decoded_words)

def main(input_tensor):
    hidden_size = 128
    max_length = 50
    
    root_dir = Path.cwd().parent
    name = 'WikiHow'
    dataset_dir = os.path.join(root_dir, 'data', name)
    save_dir = os.path.join(root_dir, 'checkpoints', name)
    
    with open(os.path.join(dataset_dir, 'feature_tokenizer.pickle'), 'rb') as handle:
        feature_tokenizer = pickle.load(handle)
        
    num_words_text = max(feature_tokenizer.word2index.values()) + 1
        
    encoder = EncoderRNN(num_words_text, hidden_size).to(torch.device('cpu'))
    decoder = AttnDecoderRNN(hidden_size, num_words_text, max_length).to(torch.device('cpu'))
    checkpoint = torch.load(os.path.join(save_dir, 'best_checkpoint.tar'), map_location=torch.device('cpu'))
    encoder.load_state_dict(checkpoint['en'])
    decoder.load_state_dict(checkpoint['de'])

    # Tokenize the input tensor
    input_tensor = feature_tokenizer.texts_to_sequences([input_tensor])
    print("input_tensor idx:", input_tensor)
    summary = summarize_on_cpu(input_tensor, encoder, decoder,
                            EOS_token=feature_tokenizer.word2index['EOS'],
                            index2word=feature_tokenizer.index2word)

    print("Summary:", summary)

if __name__ == "__main__":
    # Ask for input tensor from the user
    input_tensor = input("Enter the input tensor ")
    main(input_tensor)
    


