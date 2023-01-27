import torch
from torch import nn
from torchtext.vocab import build_vocab_from_iterator
from vocab_dataloader import loader_for_vocab
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LSTMModel(nn.Module):

    def __init__(self, vocab_size, embed_dim, dropout, dropout_lstm, num_hidden_layers, size_hidden_layer, classes):
        super(LSTMModel, self).__init__()
        self.size_embed = embed_dim
        self.size_hidden_layer = size_hidden_layer
        self.num_hidden_layers = num_hidden_layers
        self.dropout_p = dropout
        self.dropout_lstm_p = dropout_lstm
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(self.vocab_size, self.size_embed, sparse=False)

        self.rnn = nn.LSTM(input_size=self.size_embed,
                           hidden_size=self.size_hidden_layer,
                           num_layers=self.num_hidden_layers,
                           bidirectional=True,
                           dropout=self.dropout_lstm_p,
                           batch_first=True)

        self.out = nn.Linear(self.size_hidden_layer, classes)

        self.attn_combine = nn.Linear(self.size_hidden_layer * 4, self.size_hidden_layer)
        self.attn = nn.Linear(2 * self.size_hidden_layer, 2 * self.size_hidden_layer)  # Attention

        self.dropout = nn.Dropout(p=self.dropout_p)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.out.weight.data.uniform_(-initrange, initrange)
        self.out.bias.data.zero_()
        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, text, batch_size):
        hidden = (self.initHidden(batch_size), self.initHidden(batch_size))
        embedded = self.embedding(text)  # Get embedding
        embedded = self.dropout(embedded)  # Apply dropout

        output, hidden = self.rnn(embedded, hidden)  # Apply LSTM

        final_hidden = hidden[0].view(self.num_hidden_layers, 2, batch_size, self.size_hidden_layer)[-1]
        final_hidden = torch.cat((final_hidden[0], final_hidden[1]), 1)  # Used for attention

        attn_weights = self.attn(output)
        attn_weights = torch.bmm(attn_weights, final_hidden.unsqueeze(2))
        attn_weights = F.softmax(attn_weights.squeeze(2), dim=1)  # Calculate attention weights

        context = torch.bmm(output.transpose(1, 2), attn_weights.unsqueeze(2)).squeeze(2)  # Calculate context

        output = self.attn_combine(torch.cat((context, final_hidden), 1))  # Combine context with final hidden layer

        output = self.out(output.reshape(batch_size, 1, -1))  # Apply final fully connected layer

        return F.log_softmax(output, dim=0)  # log_softmax because Cross Entropy is our loss function

    def initHidden(self, batch_size):
        return torch.zeros(2 * self.num_hidden_layers, batch_size, self.size_hidden_layer, device=device)


class VocabSizes():
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.max_len = 0
        self.label_dict = {}
        self.loader = loader_for_vocab()

    @staticmethod
    def yield_tokens_text(self, data_iter):
        count = 0
        for label, text in data_iter:
            if label[0] not in self.label_dict.keys():
                self.label_dict[label[0]] = count
                count += 1
            the_tokenized = self.tokenizer(text[0])
            if len(the_tokenized) > self.max_len:
                self.max_len = len(the_tokenized)
            yield the_tokenized

    def get_vocab_size_text(self):
        train_iter = iter(self.loader)
        vocab_text = build_vocab_from_iterator(self.yield_tokens_text(self, train_iter), specials=["<unk>"])
        vocab_text.set_default_index(vocab_text["<unk>"])
        return len(vocab_text), vocab_text

    def get_max_len(self):
        return self.max_len

    def get_label_dict(self):
        return self.label_dict

    def get_int_to_label_dict(self):
        return {v: k for k, v in self.label_dict.items()}
