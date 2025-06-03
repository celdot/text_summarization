import streamlit as st
import torch
import os
import pickle
import re
from pathlib import Path

from utils.models import EncoderRNN, EncoderRNN_packed, AttnDecoderRNN, AttnDecoderRNN_packed
from utils.inference import summarize_on_cpu

# ------------------------- Constants -------------------------
MAX_INPUT_LEN = 400  # You control this; user cannot change it

# ------------------------- Sidebar Configuration -------------------------
st.sidebar.header("Configuration")

root_dir = st.sidebar.text_input("Root Directory", value=str(Path.cwd().parent))
name = st.sidebar.selectbox("Dataset Name", options=["WikiHow"], index=0)
checkpoint_name = st.sidebar.text_input("Checkpoint Name", value="best_checkpoint.tar")
legacy = st.sidebar.checkbox("Use Legacy (non-packed) Model", value=False)
hidden_size = st.sidebar.slider("Hidden Size", min_value=64, max_value=512, value=128, step=32)
max_output_len = st.sidebar.slider("Max Summary Output Length", min_value=10, max_value=300, value=50, step=10)

# ------------------------- Model + Tokenizer Loading -------------------------
@st.cache_resource
def load_all(root_dir, name, checkpoint_name, hidden_size, max_output_len, legacy):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if legacy:
        dataset_dir = os.path.join(root_dir, 'data', name)
        save_dir = os.path.join(root_dir, 'checkpoints', name)
    else:
        dataset_dir = os.path.join(root_dir, 'data_packed', name)
        save_dir = os.path.join(root_dir, 'checkpoints_packed', name)

    with open(os.path.join(dataset_dir, 'feature_tokenizer.pickle'), 'rb') as handle:
        feature_tokenizer = pickle.load(handle)

    num_words_text = max(feature_tokenizer.word2index.values()) + 1

    if legacy:
        encoder = EncoderRNN(num_words_text, hidden_size).to(device)
        decoder = AttnDecoderRNN(hidden_size, num_words_text, max_output_len).to(device)
    else:
        encoder = EncoderRNN_packed(num_words_text, hidden_size).to(device)
        decoder = AttnDecoderRNN_packed(hidden_size, num_words_text, max_output_len).to(device)

    checkpoint_path = os.path.join(save_dir, checkpoint_name)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    encoder.load_state_dict(checkpoint['en'])
    decoder.load_state_dict(checkpoint['de'])

    encoder.to(device).eval()
    decoder.to(device).eval()

    return encoder, decoder, feature_tokenizer, device

encoder, decoder, tokenizer, device = load_all(
    root_dir, name, checkpoint_name, hidden_size, max_output_len, legacy
)

# ------------------------- Utility Functions -------------------------
def convert_input(text, tokenizer):
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text).lower().strip()
    token_ids = tokenizer.texts_to_sequences([text])[0]
    return token_ids

# ------------------------- Streamlit UI -------------------------
st.title("Text Summarizer (GRU Encoder-Decoder)")

input_method = st.radio("Choose input method", ["Type text", "Upload .txt file"])
user_text = ""

if input_method == "Type text":
    user_text = st.text_area("Enter text to summarize", height=200)
elif input_method == "Upload .txt file":
    uploaded_file = st.file_uploader("Upload a text file", type=["txt"])
    if uploaded_file is not None:
        user_text = uploaded_file.read().decode("utf-8")

if st.button("Summarize") and user_text:
    try:
        input_ids = convert_input(user_text, tokenizer)

        if len(input_ids) > MAX_INPUT_LEN:
            st.warning(f"Input is too long ({len(input_ids)} tokens). Limit is {MAX_INPUT_LEN}. Please shorten your text.")
        else:
            summary = summarize_on_cpu(
                input_ids,
                encoder,
                decoder,
                EOS_token=tokenizer.word2index['EOS'],
                index2word=tokenizer.index2word,
                legacy=legacy
            )
            st.subheader("Summary")
            st.success(summary)
    except Exception as e:
        st.error(f"An error occurred: {e}")
