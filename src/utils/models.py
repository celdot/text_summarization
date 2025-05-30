import torch
import torch.nn as nn
import torch.nn.functional as F

SOS_token = 1 # Start-of-sentence token
EOS_token = 2 # End-of-sentence token
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
LSTM with attention
"""
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_p=0.1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout_p)

        self.output_projection = nn.Linear(2*self.hidden_size, hidden_size)

    def forward(self, input):
        # Input is of size (B,seq_length)
        embedded = self.dropout(self.embedding(input)) # embedded: (B,seq_length, embedding_dim) => here embedding_dim = H
        output, hidden = self.gru(embedded)  # hidden: (2, B, H), output (B, seq_length, 2*H)

        # Combine the two directions
        # hidden = hidden[0:hidden.size(0):2] + hidden[1:hidden.size(0):2]  # (1, B, H)
        # Or alternatively use linear projection over average of the hiddenstates + tanh-activation
        mean_hidden = output.mean(dim=1) # (B,2*H)

        hidden = torch.tanh(self.output_projection(mean_hidden)) # (B,H)

        hidden = hidden.unsqueeze(0) # (1,B,H)

        return output, hidden


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(SOS_token)
        decoder_hidden = encoder_hidden
        decoder_outputs = []

        for i in range(max_length):
            decoder_output, decoder_hidden  = self.forward_step(decoder_input, decoder_hidden)
            decoder_outputs.append(decoder_output)

            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1) # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()  # detach from history as input

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        return decoder_outputs, decoder_hidden, None # We return `None` for consistency in the training loop

    def forward_step(self, input, hidden):
        output = self.embedding(input)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.out(output)
        return output, hidden

class DotAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, encoder_hidden_states, decoder_hidden_state):
        # encoder_hidden_states: (B, T, H) # with T == S from perviously and == seq_length of encoder
        # decoder_hidden_state:  (B, H)

        # Compute attention scores
        attention_weights = torch.bmm(encoder_hidden_states, decoder_hidden_state.unsqueeze(2))  # (B, T, 1)
        attention_weights = F.softmax(attention_weights.squeeze(2), dim=1)  # (B, T)

        # Compute context vector
        context = torch.bmm(encoder_hidden_states.transpose(1, 2), attention_weights.unsqueeze(2)).squeeze(2)  # (B, H)

        # Concatenate context with decoder hidden state
        return torch.cat((context, decoder_hidden_state), dim=1), attention_weights

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, max_length, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.max_length = max_length

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.encoder_projection = nn.Linear(2 * hidden_size, hidden_size)
        self.attention = DotAttention()
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.context_hidden = nn.Linear(2*hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        '''Computes the forward pass of the Decoder network

        Args:
            encoder_outputs: size (B,S,2*H)
            encoder_hidden: size (1,B,H)
            target_tensor: (B, T) # T== seq_length in this function
            '''
        batch_size = encoder_outputs.size(0)
        seq_len = target_tensor.size(1) if target_tensor is not None else self.max_length

        # Form the initial decoder input which consists of the start tokens concetanted with the 
        decoder_input = torch.full((batch_size, 1), SOS_token, dtype=torch.long, device=device)
        decoder_hidden = encoder_hidden
        decoder_outputs = []
        attentions = []

        if target_tensor is not None:
            # Teacher forcing: embed all targets at once
            inputs = self.embedding(target_tensor)  # (B, T, H)
            inputs = self.dropout(inputs)

            outputs, hidden = self.gru(inputs, decoder_hidden)  # (B, T, H), (1, B, H)

            # Project encoder outputs
            projected_encoder_outputs = self.encoder_projection(encoder_outputs)  # (B, S, H)

            # Attention scores: batched dot product
            attn_scores = torch.bmm(projected_encoder_outputs, outputs.transpose(1, 2))  # (B, S, T)
            attn_weights = F.softmax(attn_scores, dim=1)  # (B, S, T)

            # Context: weighted sum
            context = torch.bmm(projected_encoder_outputs.transpose(1, 2), attn_weights)  # (B, H, T)
            context = context.transpose(1, 2)  # (B, T, H)

            # Concatenate context and decoder output
            combined = torch.cat((context, outputs), dim=2)  # (B, T, 2H)

            hidden_contextualized = torch.tanh(self.context_hidden(combined)) # (B,T,2H)
            output = self.out(hidden_contextualized)  # (B, T, V)
            output = F.log_softmax(output, dim=-1)

            return output, hidden, attn_weights.transpose(1, 2)  # (B, T, V), (1, B, H), (B, T, S)

        else:
            # Inference: step by step
            for _ in range(self.max_length):
                decoder_output, decoder_hidden, attn_weights = self.forward_step(
                    decoder_input, decoder_hidden, encoder_outputs
                )
                decoder_outputs.append(decoder_output)
                attentions.append(attn_weights)

                decoder_input = decoder_output.argmax(dim=-1)

            decoder_outputs = torch.cat(decoder_outputs, dim=1)
            decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
            attention_scores = torch.cat(attentions, dim=1)

            return decoder_outputs, decoder_hidden, attention_scores
