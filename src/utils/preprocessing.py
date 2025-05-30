import os

import matplotlib.pyplot as plt
import numpy as np


def preprocess_text(df, stopwords, column_name):
    # Select only alphanumeric characters
    df[column_name] = df[column_name].str.replace(r'[^a-zA-Z0-9\s]', '', regex=True)
    
    # Convert to lowercase
    df[column_name] = df[column_name].str.lower()
    
    # Strip leading and trailing whitespace
    df[column_name] = df[column_name].str.strip()

    # Remove stopwords
    df[column_name] = df[column_name].apply(lambda x: ' '.join([word for word in x.split() if word not in stopwords]))
    
    # Remove empty strings
    df[column_name] = df[column_name].replace('', np.nan)
    df = df.dropna(subset=[column_name])
    
    return df

def add_start_end_tokens(df, column_name, start_token, end_token):
    df[column_name] = start_token + df[column_name] + end_token
    return df

def get_length(df, column_name):
    df[f'length_{column_name}'] = df[column_name].apply(lambda x: len(x.split()))
    return df

def preprocess_data(df, stopwords, column_name, start_token, end_token):
    df = preprocess_text(df, stopwords, column_name)
    df = add_start_end_tokens(df, column_name, start_token, end_token)
    df = get_length(df, column_name)
    return df

# Get length where k percent of the data is below
def get_length_at_percentile(df, k, column_name):
    length_at_k = df[column_name].quantile(k)
    return length_at_k

def save_data(df, output_dir, filename):
    output_path = os.path.join(output_dir, f"{filename}.csv")
    df.to_csv(output_path, index=False)

def preprocessing_pipeline(df, stopwords, output_dir, filename, subset_size = 0.2, start_token='SOS ', end_token=' EOS'):
    # Select relevant columns
    if filename == "wikihow_data":
        df = df[['headline', 'text']]
        df = df.rename(columns={"headline": "summary"})

    # Get a random sample of the dataset
    df = df.sample(frac=subset_size, random_state=42)
    
    # Drop rows with NaN values
    df = df.dropna()
    
    # Preprocess the 'summary' column
    df = preprocess_data(df, stopwords, 'summary', start_token, end_token)
    
    # Preprocess the 'text' column
    df = preprocess_data(df, stopwords, 'text', start_token, end_token)
    
    # Keep only the rows where the length of the text and summary are below the 95th percentile
    length_summary_95 = get_length_at_percentile(df, 0.99, 'length_summary')
    length_text_95 = get_length_at_percentile(df, 0.99, 'length_text')
    df = df[(df['length_summary'] <= length_summary_95) & (df['length_text'] <= length_text_95)]
    
    save_data(df, output_dir, filename)

    del df

def get_data_distribution(df, figures_dir, dataset_name):
    """
    Plot the distribution of the text and summary lengths in the dataset.
    """
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    text_word_count = []
    headline_word_count = []

    for i in df['text']:
      text_word_count.append(len(i.split()))

    for i in df['summary']:
      headline_word_count.append(len(i.split()))

    ax[0].hist(text_word_count, bins=50)
    ax[0].set_title("Text")
    ax[0].set_xlabel("Word Count")
    ax[0].set_ylabel("Frequency")
    ax[1].hist(headline_word_count, bins=50)
    ax[1].set_title("Summary")
    ax[1].set_xlabel("Word Count")
    ax[1].set_ylabel("Frequency")
    plt.suptitle("Distribution of Text and Summary Lengths")
    plt.savefig(os.path.join(figures_dir, f"{dataset_name}_text_summary_length_distribution.png"))
    plt.show()
    plt.close(fig)
    