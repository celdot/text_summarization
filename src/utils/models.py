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

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, max_length, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.max_length = max_length

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.encoder_projection = nn.Linear(2 * hidden_size, hidden_size)
        self.gru = nn.GRU(2*hidden_size, hidden_size, batch_first=True)
        self.context_hidden = nn.Linear(2*hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_p)

        
    def attention(self, encoder_outputs, decoder_hidden):
        '''Calculates the attention mechanism
        Args:
            encoder_outputs: size (B,S,2*H)
            decoder_hidden: size (1,B,H)
        '''
         # Project encoder outputs
        projected_encoder_outputs = self.encoder_projection(encoder_outputs)  # (B, S, H)

        # Attention scores: batched dot product (decoder_hidden = (1,B,H))
        attn_scores = torch.bmm(projected_encoder_outputs, decoder_hidden.permute(1, 2, 0))  # (B, S, 1)
        attn_weights = F.softmax(attn_scores, dim=1)  # (B, S, 1)

        # Context: weighted sum
        context = torch.bmm(projected_encoder_outputs.transpose(1, 2), attn_weights)  # (B, H, 1)
        context = context.transpose(1, 2)  # (B, 1, H)

        # Concatenate context and decoder output (which is equal to the final hidden state in the step-by-step approach) (decoder_hidden = (1,B,H))
        combined = torch.cat((context, decoder_hidden.permute(1, 0, 2)), dim=2)  # (B, 1, 2H)

        decoder_hidden_contextualized = torch.tanh(self.context_hidden(combined)) # (B,1,H)

        return decoder_hidden_contextualized, attn_weights
    
    def forward_step(self, decoder_input, decoder_hidden, decoder_hidden_contextualized, encoder_outputs):
        '''Implements a single forward step of the deocder network with attention
        Args:
            decoder_input: size (B, 1)
            decoder_hidden: size (1, B, H)
            decoder_hidden_contextualized: size (B,1,H)
            encoder_outputs: size (B,S,2*H)
        '''
        
        # 1. Embed the decoder input and concatenate the contextualized hidden
        inputs = self.embedding(decoder_input)
        inputs = torch.cat((inputs, decoder_hidden_contextualized), dim=2) # (B,1,2H)
        inputs = self.dropout(inputs)

        _, hidden = self.gru(inputs, decoder_hidden)  # (B, 1, H), (1, B, H)

        hidden_contextualized, attn_weights = self.attention(encoder_outputs, hidden)
        
        output = self.out(hidden_contextualized)  # (B, 1, V)

        return output, hidden, hidden_contextualized, attn_weights # (B,1,V), (1,B,H), (B,1,H), (B,S,1)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        '''Computes the forward pass of the Decoder network

        Args:
            encoder_outputs: size (B,S,2*H)
            encoder_hidden: size (1,B,H)
            target_tensor: (B, T) # T== seq_length in this function
            '''
        batch_size = encoder_outputs.size(0)
        T = target_tensor.size(1) if target_tensor is not None else self.max_length

        # Form the initial decoder input which consists of the start tokens concetanted with the 
        decoder_input = torch.full((batch_size, 1), SOS_token, dtype=torch.long, device=device)
        decoder_hidden = encoder_hidden # (1,B,H)

        # Initialize the first contextualized hidden state
        decoder_hidden_contextualized, _ = self.attention(encoder_outputs, decoder_hidden)
        
        decoder_outputs = []
        attentions = []

        for t in range(T):
            decoder_output, decoder_hidden, decoder_hidden_contextualized, attn_weights = self.forward_step(
                decoder_input, decoder_hidden, decoder_hidden_contextualized, encoder_outputs
            )  # (B,1,V), (1,B,H), (B,1,H), (B,S,1)
            decoder_outputs.append(decoder_output)
            attentions.append(attn_weights)

            # Use teacher forcing
            if target_tensor is not None:
                decoder_input = target_tensor[:, t].unsqueeze(1)  # (B,1)
            else:
                decoder_input = decoder_output.argmax(dim=-1)  # (B, 1)
    
        decoder_outputs = torch.cat(decoder_outputs, dim=1)  #(B,T,V)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        attention_scores = torch.cat(attentions, dim=-1) # (B,T, S)

        return decoder_outputs, decoder_hidden, attention_scores.transpose(1, 2)  # (B, T, V), (1, B, H), (B, T, S)