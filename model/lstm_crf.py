import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM_CRF(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size, pretrained_embedding = None):
        super(LSTM_CRF, self).__init__()
        self.hidden_dim = hidden_dim

        if pretrained_embedding == None:
            # generate random word_vec
            self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        else:
            self.word_embeddings = nn.Embedding.from_pretrained(pretrained_embedding)
        # print(self.word_embeddings)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        # print(sentence)
        embeds = self.word_embeddings(sentence)
        # print(embeds)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores


class LSTM_CRF_(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size, start_tag_ix, stop_tag_ix, pretrained_embedding = None):
        super(LSTM_CRF, self).__init__()
        self.hidden_dim = hidden_dim

        if pretrained_embedding == None:
            # generate random word_vec
            self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        else:
            self.word_embeddings = nn.Embedding.from_pretrained(pretrained_embedding)
        # print(self.word_embeddings)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

        # transition matrix
        self.transitions = nn.Parameter(torch.randn(self.tagset_size,self.tagset_size))
        # restrict
        for i in start_tag_ix:
            self.transitions.data[i, :] = -10000    # can not translate to start_tag
        for i in stop_tag_ix:
            self.transitions.data[:, i] = -10000    # can not translate from stop_tag

    def forward(self, sentence):
        # print(sentence)
        embeds = self.word_embeddings(sentence)
        # print(embeds)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores

