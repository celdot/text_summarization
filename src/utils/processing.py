import os
import pickle

import numpy as np
import pandas as pd
import tensorflow as tf
import torch
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences
from torch.utils.data import DataLoader, RandomSampler, TensorDataset


def tokenize_data(X_train, X_val, X_test, y_train, y_val, y_test, dataset_dir, max_features, load_tokenizer = False):
    """
    Tokenizes the data using Keras Tokenizer and converts the text to sequences.
    """
    if load_tokenizer:
        with open(os.path.join(dataset_dir,'feature_tokenizer.pickle'), 'rb') as handle:
            feature_tokenizer = pickle.load(handle)
    else:
        feature_tokenizer = Tokenizer(num_words=max_features)
        feature_tokenizer.fit_on_texts(X_train)
        # Switch the index of SOS and EOS tokens
        temp_word_1 = feature_tokenizer.index_word[1]
        temp_word_2 = feature_tokenizer.index_word[2]
        feature_tokenizer.index_word[1] = "SOS"
        feature_tokenizer.index_word[2] = "EOS"
        feature_tokenizer.word_index["SOS"] = 1
        feature_tokenizer.word_index["EOS"] = 2
        nb_word_feature = len(feature_tokenizer.index_word)
        feature_tokenizer.index_word[nb_word_feature+1] = temp_word_1
        feature_tokenizer.index_word[nb_word_feature+2] = temp_word_2
        feature_tokenizer.word_index[temp_word_1] = nb_word_feature+1
        feature_tokenizer.word_index[temp_word_2] = nb_word_feature+2
        # Save the tokenizer with pickle
        with open(os.path.join(dataset_dir,'feature_tokenizer.pickle'), 'wb') as handle:
            pickle.dump(feature_tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # Save vocabulary dictionary with pickle
        with open(os.path.join(dataset_dir,'feature_vocab.pickle'), 'wb') as handle:
            pickle.dump(feature_tokenizer.index_word, handle, protocol=pickle.HIGHEST_PROTOCOL)

    X_train = feature_tokenizer.texts_to_sequences(X_train)
    X_val = feature_tokenizer.texts_to_sequences(X_val)
    X_test = feature_tokenizer.texts_to_sequences(X_test)

    print("Number of Samples in X_train:", len(X_train))

    if load_tokenizer:
        with open(os.path.join(dataset_dir,'label_tokenizer.pickle'), 'rb') as handle:
            label_tokenizer = pickle.load(handle)
    else:
        label_tokenizer = Tokenizer(num_words=max_features)
        label_tokenizer.fit_on_texts(y_train)
        # Switch the index of SOS and EOS tokens
        temp_word_1 = label_tokenizer.index_word[1]
        temp_word_2 = label_tokenizer.index_word[2]
        label_tokenizer.index_word[1] = "SOS"
        label_tokenizer.index_word[2] = "EOS"
        label_tokenizer.word_index["SOS"] = 1
        label_tokenizer.word_index["EOS"] = 2
        nb_word_label = len(label_tokenizer.index_word)
        label_tokenizer.index_word[nb_word_label+1] = temp_word_1
        label_tokenizer.index_word[nb_word_label+2] = temp_word_2
        label_tokenizer.word_index[temp_word_1] = nb_word_label+1
        label_tokenizer.word_index[temp_word_2] = nb_word_label+2
        # Save tokenizer with pickle
        with open(os.path.join(dataset_dir,'label_tokenizer.pickle'), 'wb') as handle:
            pickle.dump(label_tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # Save vocabulary dictionary with pickle
        with open(os.path.join(dataset_dir,'label_vocab.pickle'), 'wb') as handle:
            pickle.dump(label_tokenizer.index_word, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
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

def processing_pipeline(dataset_dir, name, batch_size = 32, num_workers = 2, max_features = 15000, load_tokenizer = False):
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
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Convert to torch tensors
    train_data = TensorDataset(torch.LongTensor(X_train).to(device), torch.LongTensor(y_train).to(device))
    val_data = TensorDataset(torch.LongTensor(X_val).to(device), torch.LongTensor(y_val).to(device))
    test_data = TensorDataset(torch.LongTensor(X_test).to(device), torch.LongTensor(y_test).to(device))
    
    # Create samplers
    train_sampler = RandomSampler(train_data)
    val_sampler = RandomSampler(val_data)
    test_sampler = RandomSampler(test_data)
    
    # Create dataloaders
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size, num_workers=num_workers)
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size, num_workers=num_workers)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size, num_workers=num_workers)
    
    # Save dataloaders
    with open(os.path.join(dataset_dir, 'train_dataloader.pickle'), 'wb') as handle:
        pickle.dump(train_dataloader, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(dataset_dir, 'val_dataloader.pickle'), 'wb') as handle:
        pickle.dump(val_dataloader, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(dataset_dir, 'test_dataloader.pickle'), 'wb') as handle:
        pickle.dump(test_dataloader, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    return train_dataloader, val_dataloader, test_dataloader, num_words_text, num_words_summary