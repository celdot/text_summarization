import os
import pickle

import numpy as np
import pandas as pd
import tensorflow as tf
import torch
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences
from torch.utils.data import DataLoader, RandomSampler, TensorDataset


def enforce_special_tokens(tokenizer, sos_token="SOS", eos_token="EOS"):
    """
    Ensure that SOS and EOS tokens are set to index 1 and 2 in both word_index and index_word.
    Moves any existing tokens that occupy 1 and 2 to new indices.
    """
    # Save any existing tokens at positions 1 and 2
    old_1 = tokenizer.index_word.get(1)
    old_2 = tokenizer.index_word.get(2)

    # Assign SOS and EOS
    tokenizer.word_index[sos_token] = 1
    tokenizer.word_index[eos_token] = 2
    tokenizer.index_word[1] = sos_token
    tokenizer.index_word[2] = eos_token

    # If something was overwritten, move it to new indices
    max_index = max(tokenizer.word_index.values())
    if old_1 and old_1 != sos_token:
        max_index += 1
        tokenizer.word_index[old_1] = max_index
        tokenizer.index_word[max_index] = old_1
    if old_2 and old_2 != eos_token:
        max_index += 1
        tokenizer.word_index[old_2] = max_index
        tokenizer.index_word[max_index] = old_2

    return tokenizer


def tokenize_data(X_train, X_val, X_test, y_train, y_val, y_test, dataset_dir, max_features, load_tokenizer=False):
    """
    Tokenizes the data using Keras Tokenizer and converts the text to sequences.
    Ensures SOS and EOS tokens are correctly assigned.
    """
    sos_token = "SOS"
    eos_token = "EOS"

    if load_tokenizer:
        with open(os.path.join(dataset_dir, 'feature_tokenizer.pickle'), 'rb') as f:
            feature_tokenizer = pickle.load(f)
    else:
        feature_tokenizer = Tokenizer(num_words=max_features, oov_token="<unk>")
        feature_tokenizer.fit_on_texts(X_train)
        print("SOS index:", feature_tokenizer.word_index.get("SOS"))
        print("EOS index:", feature_tokenizer.word_index.get("EOS"))
        feature_tokenizer = enforce_special_tokens(feature_tokenizer, sos_token, eos_token)
        with open(os.path.join(dataset_dir, 'feature_tokenizer.pickle'), 'wb') as f:
            pickle.dump(feature_tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(dataset_dir, 'feature_vocab.pickle'), 'wb') as f:
            pickle.dump(feature_tokenizer.index_word, f, protocol=pickle.HIGHEST_PROTOCOL)

    X_train = feature_tokenizer.texts_to_sequences(X_train)
    X_val = feature_tokenizer.texts_to_sequences(X_val)
    X_test = feature_tokenizer.texts_to_sequences(X_test)

    print("Number of Samples in X_train:", len(X_train))

    if load_tokenizer:
        with open(os.path.join(dataset_dir, 'label_tokenizer.pickle'), 'rb') as f:
            label_tokenizer = pickle.load(f)
    else:
        label_tokenizer = Tokenizer(num_words=max_features, oov_token="<unk>")
        label_tokenizer.fit_on_texts(y_train)
        print("SOS index:", label_tokenizer.word_index.get("SOS"))
        print("EOS index:", label_tokenizer.word_index.get("EOS"))
        label_tokenizer = enforce_special_tokens(label_tokenizer, sos_token, eos_token)
        with open(os.path.join(dataset_dir, 'label_tokenizer.pickle'), 'wb') as f:
            pickle.dump(label_tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(dataset_dir, 'label_vocab.pickle'), 'wb') as f:
            pickle.dump(label_tokenizer.index_word, f, protocol=pickle.HIGHEST_PROTOCOL)

    y_train = label_tokenizer.texts_to_sequences(y_train)
    y_val = label_tokenizer.texts_to_sequences(y_val)
    y_test = label_tokenizer.texts_to_sequences(y_test)

    print("Number of Samples in y_train:", len(y_train))

    return X_train, X_val, X_test, y_train, y_val, y_test, feature_tokenizer, label_tokenizer


def add_padding(X_train, X_val, X_test, y_train, y_val, y_test, maxlen_text, maxlen_summary):
    """
    Adds padding to the sequences to make them of equal length.
    """
    X_train = pad_sequences(X_train, maxlen = maxlen_text, padding='post')
    X_val = pad_sequences(X_val, maxlen = maxlen_text, padding='post')
    X_test = pad_sequences(X_test, maxlen = maxlen_text, padding='post')

    y_train = pad_sequences(y_train, maxlen = maxlen_summary, padding='post')
    y_val = pad_sequences(y_val, maxlen = maxlen_summary, padding='post')
    y_test = pad_sequences(y_test, maxlen = maxlen_summary, padding='post')

    return X_train, X_val, X_test, y_train, y_val, y_test

# Delete rows that only contain padding
def delete_padding_rows(X_train, X_val, X_test, y_train, y_val, y_test):
    """
    Deletes rows that only contain padding.
    """
    # For training set
    mask_train = ~np.all(X_train == 0, axis=1)
    X_train = X_train[mask_train]
    y_train = y_train[mask_train]
    
    # For validation set
    mask_val = ~np.all(X_val == 0, axis=1)
    X_val = X_val[mask_val]
    y_val = y_val[mask_val]

    # For test set
    mask_test = ~np.all(X_test == 0, axis=1)
    X_test = X_test[mask_test]
    y_test = y_test[mask_test]

    return X_train, X_val, X_test, y_train, y_val, y_test

def save_as_tensor(dataset_dir, data, filename):
    """
    Save the data as a torch tensor.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Saving {filename}...")
    if os.path.exists(os.path.join(dataset_dir, filename)):
        print(f"{filename} already exists. Skipping.")
        return
    data = torch.from_numpy(data).long().to(device)
    torch.save(data, os.path.join(dataset_dir, filename))

def processing_pipeline(dataset_dir, name, max_features = 15000, load_tokenizer = False):
    """
    Process the data by splitting it into train, validation, and test sets,
    tokenizing the data, and adding padding to the sequences.
    """
    df = pd.read_csv(os.path.join(dataset_dir, name + '.csv'))
    maxlen_text = df["text"].str.split().str.len().max()
    maxlen_summary = df["summary"].str.split().str.len().max()
    print("Max length of text:", maxlen_text)
    print("Max length of summary:", maxlen_summary)

    # Split the data into train, validation, and test sets
    X_train, X_middle, y_train, y_middle = train_test_split(df.text, df.summary, test_size = 0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_middle, y_middle, test_size = 0.4, random_state=42)

    del df

    # Tokenize the data
    X_train, X_val, X_test, y_train, y_val, y_test, feature_tokenizer, label_tokenizer = tokenize_data(X_train, X_val, X_test, y_train, y_val, y_test, dataset_dir, max_features, load_tokenizer)
    
    # Add 1 to the vocab size to account for padding
    num_words_text = len(feature_tokenizer.word_index) + 1 # because we are using 1-based indexing (0 is reserved for padding)
    num_words_summary = len(label_tokenizer.word_index) + 1

    # Add padding to the sequences
    X_train, X_val, X_test, y_train, y_val, y_test = add_padding(X_train, X_val, X_test, y_train, y_val, y_test,
                                                                 maxlen_text,
                                                                 maxlen_summary)

    # Delete rows that only contain padding
    X_train, X_val, X_test, y_train, y_val, y_test = delete_padding_rows(X_train,
                                                                        X_val,
                                                                        X_test,
                                                                        y_train,
                                                                        y_val,
                                                                        y_test)
    
    print("index of start and end token", feature_tokenizer.word_index["SOS"], feature_tokenizer.word_index["EOS"])
    
    save_as_tensor(dataset_dir, X_train, "x_train.pt")
    save_as_tensor(dataset_dir, X_val, "x_val.pt")
    save_as_tensor(dataset_dir, X_test, "x_test.pt")
    save_as_tensor(dataset_dir, y_train, "y_train.pt")
    save_as_tensor(dataset_dir, y_val, "y_val.pt")
    save_as_tensor(dataset_dir, y_test, "y_test.pt")
    
    return num_words_text, num_words_summary