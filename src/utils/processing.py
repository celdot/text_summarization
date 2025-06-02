import os
import pickle

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

SOS_token = 1
EOS_token = 2

class Tokenizer:
    def __init__(self):
        self.word2index = {"PAD": 0, "SOS": 1, "EOS": 2, "UNK": 3}
        self.word2count = {"PAD": 0, "SOS": 0, "EOS": 0, "UNK": 0}
        self.index2word = {0: "PAD", 1: "SOS", 2: "EOS", 3: "UNK"}
        self.n_words = 4  # Count SOS, EOS, and UNK

    def fit_on_texts(self, texts):
        """
        Fit the tokenizer on the provided texts.
        """
        for text in texts:
            self.add_sentence(text)

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1
            
    def texts_to_sequences(self, texts):
        """
        Convert a list of texts to sequences of indices.
        """
        sequences = []
        for text in texts:
            if text is None:
                continue
            sequence = [self.word2index.get(word, self.word2index.get("UNK")) for word in text.split()]
            sequences.append(sequence)
        return sequences
    
def pad_sequences(sequences, maxlen):
    padded_sequences = []
    for i, seq in enumerate(sequences):
        if not isinstance(seq, list):
            print(f"Warning: Invalid sequence at index {i}: {seq}")
            seq = []

        # Ensure elements are integers
        seq = [int(token) for token in seq]

        # Pad or truncate to match maxlen
        if len(seq) < maxlen:
            padded_seq = seq + [0] * (maxlen - len(seq))
        else:
            padded_seq = seq[:maxlen]

        padded_sequences.append(padded_seq)

    # Convert to NumPy array with explicit int64 dtype
    return np.array(padded_sequences, dtype=np.int64)

def tokenize_data(X_train, X_val, X_test, y_train, y_val, y_test, dataset_dir, load_tokenizer=False):
    """
    Tokenizes the data using custom Tokenizer and converts the text to sequences.
    Ensures SOS and EOS tokens are correctly assigned.
    """

    if load_tokenizer:
        with open(os.path.join(dataset_dir, 'feature_tokenizer.pickle'), 'rb') as f:
            feature_tokenizer = pickle.load(f)
    else:
        feature_tokenizer = Tokenizer()
        feature_tokenizer.fit_on_texts(X_train)
        feature_tokenizer.fit_on_texts(y_train)
        # Print the specific tokens
        print("SOS token index:", feature_tokenizer.word2index.get("SOS", "Not found"))
        print("EOS token index:", feature_tokenizer.word2index.get("EOS", "Not found"))
        print("UNK token index:", feature_tokenizer.word2index.get("UNK", "Not found"))
        print("PAD token index:", feature_tokenizer.word2index.get("PAD", "Not found"))
        # Save the word index mapping in a txt file
        with open(os.path.join(dataset_dir, 'word_index.txt'), 'w') as f:
            for word, index in feature_tokenizer.word2index.items():
                f.write(f"{word}\t{index}\n")
        with open(os.path.join(dataset_dir, 'feature_tokenizer.pickle'), 'wb') as f:
            pickle.dump(feature_tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL)

    X_train = feature_tokenizer.texts_to_sequences(X_train)
    X_val = feature_tokenizer.texts_to_sequences(X_val)
    X_test = feature_tokenizer.texts_to_sequences(X_test)

    print("Number of Samples in X_train:", len(X_train))
    print("Number of Samples in X_val:", len(X_val))
    print("Number of Samples in X_test:", len(X_test))

    y_train = feature_tokenizer.texts_to_sequences(y_train)
    y_val = feature_tokenizer.texts_to_sequences(y_val)
    y_test = feature_tokenizer.texts_to_sequences(y_test)

    print("Number of Samples in y_train:", len(y_train))
    print("Number of Samples in y_val:", len(y_val))
    print("Number of Samples in y_test:", len(y_test))
    
    print("Vocabulary size:", feature_tokenizer.n_words)

    return X_train, X_val, X_test, y_train, y_val, y_test


def add_padding(X_train, X_val, X_test, y_train, y_val, y_test, maxlen_text, maxlen_summary):
    """
    Adds padding to the sequences to make them of equal length.
    """
    X_train = pad_sequences(X_train, maxlen = maxlen_text)
    X_val = pad_sequences(X_val, maxlen = maxlen_text)
    X_test = pad_sequences(X_test, maxlen = maxlen_text)

    y_train = pad_sequences(y_train, maxlen = maxlen_summary)
    y_val = pad_sequences(y_val, maxlen = maxlen_summary)
    y_test = pad_sequences(y_test, maxlen = maxlen_summary)

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

    print(f"{filename}, shape: {data.shape}")

    data = torch.from_numpy(data).long().to(device)
    torch.save(data, os.path.join(dataset_dir, filename))

def processing_pipeline(dataset_dir, name, load_tokenizer = False):
    """
    Process the data by splitting it into train, validation, and test sets,
    tokenizing the data, and adding padding to the sequences.
    """
    df = pd.read_csv(os.path.join(dataset_dir, name + '.csv'))
    maxlen_text = df["text"].str.split().str.len().max()
    maxlen_summary = df["summary"].str.split().str.len().max()
    maxlen_text = max(maxlen_text, 150)  # Ensure text has a minimum length
    maxlen_summary = max(maxlen_summary, 20)  # Ensure summary has a minimum length
    print("Max length of text:", maxlen_text)
    print("Max length of summary:", maxlen_summary)

    # Split the data into train, validation, and test sets
    X_train, X_middle, y_train, y_middle = train_test_split(df.text, df.summary, test_size = 0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_middle, y_middle, test_size = 0.4, random_state=42)

    del df

    # Tokenize the data
    X_train, X_val, X_test, y_train, y_val, y_test = tokenize_data(X_train, X_val, X_test, y_train, y_val, y_test, dataset_dir, load_tokenizer)
    
    # Add padding to the sequences
    X_train, X_val, X_test, y_train, y_val, y_test = add_padding(X_train, X_val, X_test, y_train, y_val, y_test, maxlen_text, maxlen_summary)

    # Delete rows that only contain padding
    X_train, X_val, X_test, y_train, y_val, y_test = delete_padding_rows(X_train,
                                                                        X_val,
                                                                        X_test,
                                                                        y_train,
                                                                        y_val,
                                                                        y_test)

    save_as_tensor(dataset_dir, X_train, "x_train.pt")
    del X_train
    save_as_tensor(dataset_dir, X_val, "x_val.pt")
    del X_val
    save_as_tensor(dataset_dir, X_test, "x_test.pt")
    del X_test
    save_as_tensor(dataset_dir, y_train, "y_train.pt")
    del y_train
    save_as_tensor(dataset_dir, y_val, "y_val.pt")
    del y_val
    save_as_tensor(dataset_dir, y_test, "y_test.pt")
    del y_test


def save_as_tensor_list(dataset_dir, data, filename):
    '''
    Saves each element of the data as a tensor in a list
    Data: list of lists of integers (n_samples, seq_length) where seq_length can vary per sample'''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Saving {filename}...")

    if os.path.exists(os.path.join(dataset_dir, filename)):
        print(f"{filename} already exists. Skipping.")
        return

    # Transform each samples to tensors
    data_t = []
    for sample in data:
        # Ensure elements are integers
        sample = [int(token) for token in sample]

        data_t.append(torch.tensor(sample).long().to(device))
    
    
    print(f"{filename} dtype: {data_t[0].dtype}, shape: {data_t[0].shape}")



    torch.save(data_t, os.path.join(dataset_dir, filename))

def del_padded_rows_packed(X_train, y_train, X_val, y_val, X_test, y_test):
    '''Same as delete_padded_rows() but now for data that is later used for packed sequences in pytorch
    => I hardcoded PAD_ID as 0 too!!'''

    def clean(X,Y):
        '''Little helper function that actually deletes the rows that only consist of 0 (the padding ID) '''
        X_clean = []
        Y_clean = []
        
        for x, y in zip(X, Y):
            # Only append that sample/label pair of both of them have an entry different from 0
            if torch.any(torch.tensor(x)) and torch.any(torch.tensor(y)):
                X_clean.append(x)
                Y_clean.append(y)
    
        return X_clean, Y_clean
    
    X_train, y_train = clean(X_train, y_train)
    X_val, y_val = clean(X_val, y_val)
    X_test, y_test = clean(X_test, y_test)

    return X_train, y_train, X_val, y_val, X_test, y_test

def processing_pipeline_packed(dataset_dir, name, load_tokenizer = False):
    """
    This is the pipeline you have to use when working with padded_packed sequences later
    It process the data by splitting it into train, validation, and test sets,
    tokenizing the data, converts each sample into a tensor, collects all samples in a list
    and saves these lists of tensors of variable length for later. 
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

    # Tokenize the data => The outputs are lists of lists containing integers => dim (n_samples, seq_lengths)
    X_train, X_val, X_test, y_train, y_val, y_test = tokenize_data(X_train, X_val, X_test, y_train, y_val, y_test, dataset_dir, load_tokenizer)
    
    # Delete samples/labels that have only padding at either samples or labels
    X_train, y_train, X_val, y_val, X_test, y_test = del_padded_rows_packed(X_train, y_train, X_val, y_val, X_test, y_test)

    # Save the lists as lists of tensors
    save_as_tensor_list(dataset_dir, X_train, "x_train_list.pt")
    del X_train
    save_as_tensor_list(dataset_dir, X_val, "x_val_list.pt")
    del X_val
    save_as_tensor_list(dataset_dir, X_test, "x_test_list.pt")
    del X_test
    save_as_tensor_list(dataset_dir, y_train, "y_train_list.pt")
    del y_train
    save_as_tensor_list(dataset_dir, y_val, "y_val_list.pt")
    del y_val
    save_as_tensor_list(dataset_dir, y_test, "y_test_list.pt")
    del y_test