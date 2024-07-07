import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    def __init__(self, n_dim, att_mask=False, device='cpu'):
        super(ScaledDotProductAttention, self).__init__()
        
        self.device = device
        self.n_dim = n_dim
        self.att_mask = att_mask
        
        self.query_model = nn.Linear(self.n_dim, self.n_dim)
        self.key_model = nn.Linear(self.n_dim, self.n_dim) 
        self.value_model = nn.Linear(self.n_dim, self.n_dim)
         
    def forward(self, query, key, value, token_ids):
        query = self.query_model(query)
        key = self.key_model(key)
        value = self.value_model(value)
        
        attention_score = torch.matmul(query, key.transpose(-2, -1))
        masked_attention_score = self.making_padding_mask(attention_score, token_ids)
        
        if self.att_mask:
            masked_attention_score = self.making_att_mask(masked_attention_score)
        
        scaled_attention_score = masked_attention_score / torch.sqrt(torch.FloatTensor([self.n_dim])).to(self.device)
        attention_weight = F.softmax(scaled_attention_score, dim=-1)
        attention_output = torch.matmul(attention_weight, value)
        return attention_output
        
    def making_padding_mask(self, attention_score, token_ids):
        padding_pos = token_ids == 0
        attention_score = attention_score.masked_fill(padding_pos.unsqueeze(1) , float('-inf'))
        return attention_score

    def making_att_mask(self, attention_score):
        n_seq = attention_score.size(1)
        mask = torch.triu(torch.ones(n_seq, n_seq), diagonal=1).to(self.device)
        attention_score = attention_score.masked_fill(mask == 1, float('-inf'))
        return attention_score
    

class MultiHeadAttention(nn.Module):
    def __init__(self, n_dim, n_head, att_mask=False, device='cpu'):
        super().__init__()
        
        self.device = device
        self.att_mask = att_mask
        self.n_dim = n_dim
        self.n_head = n_head

        self.attention_list = nn.ModuleList([ScaledDotProductAttention(n_dim//n_head, att_mask, device) for _ in range(n_head)])
        self.linear = nn.Linear(n_dim, n_dim)
        
    def forward(self, query, key, value, token_ids):
        attention_outputs = []
        head_index = 0
        head_dim = self.n_dim // self.n_head
        for attention in self.attention_list:
            head_qury = query[:,:,head_index: head_index+head_dim]
            head_key = key[:,:,head_index: head_index+head_dim]
            head_value = value[:,:,head_index: head_index+head_dim]
            attention_output = attention(head_qury, head_key, head_value, token_ids)
            attention_outputs.append(attention_output)
            head_index += self.n_head
            
        concat_attention_output = torch.cat(attention_outputs, dim=-1)
        output = self.linear(concat_attention_output)
        
        return output
    

class FeedForwardNet(nn.Module):
    def __init__(self, n_dim, device='cpu'):
        super(FeedForwardNet, self).__init__()
        
        self.device = device
        self.n_dim = n_dim
        self.n_hidden = n_dim * 4
        
        self.linear1 = nn.Linear(self.n_dim, self.n_hidden)
        self.linear2 = nn.Linear(self.n_hidden, self.n_dim)
        
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x
    
    
class EncoderLayer(nn.Module):
    def __init__(self, n_dim, n_head, device='cpu'):
        super(EncoderLayer, self).__init__()
        
        self.device = device
        self.n_dim = n_dim
        self.n_head = n_head
        
        self.multi_head_attention = MultiHeadAttention(n_dim, n_head, device=device)
        self.feed_forward_net = FeedForwardNet(n_dim, device)
        
    def forward(self, x, token_ids):
        n_seq = x.size(1)
        attention_output = self.multi_head_attention(x, x, x, token_ids)        
        attention_output = F.layer_norm(attention_output + x, normalized_shape=[n_seq, self.n_dim])
        
        
        feed_forward_output = self.feed_forward_net(attention_output)
        output = F.layer_norm(feed_forward_output + attention_output, normalized_shape=[n_seq, self.n_dim])
        
        return output


class DecoderLayer(nn.Module):
    def __init__(self, n_dim, n_head, device='cpu'):
        super(DecoderLayer, self).__init__()
        
        self.device = device
        self.n_dim = n_dim
        self.n_head = n_head
        
        self.masked_multi_head_attention = MultiHeadAttention(n_dim, n_head, att_mask=True, device=device)
        self.encoder_decoder_attention = MultiHeadAttention(n_dim, n_head, device=device)
        self.feed_forward_net = FeedForwardNet(n_dim, device)
        
    def forward(self, x, encoder_output, en_token_ids, de_token_ids):
        n_seq = x.size(1)
        masked_attention_output = self.masked_multi_head_attention(x, x, x, de_token_ids)
        masked_attention_output = F.layer_norm(masked_attention_output + x, normalized_shape=[n_seq, self.n_dim])
        
        encoder_decoder_attention_output = self.encoder_decoder_attention(masked_attention_output, encoder_output, encoder_output, en_token_ids)
        encoder_decoder_attention_output = F.layer_norm(encoder_decoder_attention_output + masked_attention_output, normalized_shape=[n_seq, self.n_dim])
        
        feed_forward_output = self.feed_forward_net(encoder_decoder_attention_output)
        output = F.layer_norm(feed_forward_output + encoder_decoder_attention_output, normalized_shape=[n_seq, self.n_dim])
        
        return output
    
    
class Transformer(nn.Module):
    def __init__(self, n_seq, n_vocab, n_dim, n_head, n_layer, device='cpu'):
        super(Transformer, self).__init__()
        
        self.device = device
        self.n_seq = n_seq
        self.n_vocab = n_vocab
        self.n_dim = n_dim
        self.n_head = n_head
        self.n_layer = n_layer        
        
        self.encoder_embedding = nn.Embedding(self.n_vocab, self.n_dim).to(self.device)
        self.decoder_embedding = nn.Embedding(self.n_vocab, self.n_dim).to(self.device)
        
        self.positional_encoding = self.position_encoding(self.n_seq, self.n_dim).to(self.device)
        
        self.encoder_layers = nn.ModuleList([EncoderLayer(self.n_dim, self.n_head, device) for _ in range(self.n_layer)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(self.n_dim, self.n_head, device) for _ in range(self.n_layer)])
        
        self.fc = nn.Linear(self.n_dim, self.n_vocab)
        
    def forward(self, enc_tokens, dec_tokens):
        enc_output = self.encoder_embedding(enc_tokens) + self.positional_encoding
        dec_output = self.decoder_embedding(dec_tokens) + self.positional_encoding
        
        en_token_ids = enc_tokens
        de_token_ids = dec_tokens
        
        for encoder_layer in self.encoder_layers:
            enc_output = encoder_layer(enc_output, en_token_ids)
            
        for decoder_layer in self.decoder_layers:
            dec_output = decoder_layer(dec_output, enc_output, en_token_ids, de_token_ids)
            
        output = self.fc(dec_output)
        return output
    
    def position_encoding(self, n_seq, n_embd):
        pos_table = []
        for pos in range(n_seq):
            pos_table.append([pos/np.power(10000, (2*(i_embd//2))/n_embd) for i_embd in range(n_embd)])
        pos_table = np.array(pos_table)
        pos_table[:, 0::2] = np.sin(pos_table[:, 0::2])
        pos_table[:, 1::2] = np.cos(pos_table[:, 1::2])
        pos_table = torch.FloatTensor(pos_table)
        return pos_table