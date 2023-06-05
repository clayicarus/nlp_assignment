import time
import torch
import torch.nn as nn
import numpy as np
import os
from dataset import LSTM_CRF_Dataset
from torch import optim
from model.lstm_crf import *
import config
import utils

a, b = utils.read_cluener_raw_dataset(config.cluener_train_set)
dataset = LSTM_CRF_Dataset(a, b, "data/cluener/vectors.txt")
training_data = dataset.get_dataset()
word_to_ix, tag_to_ix = dataset.get_ix_dict()
em = torch.from_numpy(np.array(dataset.get_word_vectors())) 
print("training_data size: {}".format(len(training_data)))
print("word_to_ix size: {}".format(len(word_to_ix)))
print("tag_to_ix size: {}".format(len(tag_to_ix)))
print("embedding size: {}".format(em.size()))

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using cuda, {}".format(torch.cuda.get_device_properties(device)))

model = LSTM_CRF(config.embedding_dim, config.hidden_dim, len(word_to_ix), len(tag_to_ix), em).to(device)
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=config.lr)

# See what the scores are before training
# Note that element i,j of the output is the score for tag j for word i.
# Here we don't need to train, so the code is wrapped in torch.no_grad()
# with torch.no_grad():
#     inputs = prepare_sequence(training_data[0][0], word_to_ix)
#     tag_scores = model(inputs)
#     print(tag_scores)

epoch = 0
if os.path.exists('snapshot/latest.pth'):
    state = torch.load('snapshot/latest.pth')
    epoch = state['next_epoch']
    model.load_state_dict(state['model'])
    optimizer.load_state_dict(state['optimizer'])
    print("latest.pth loaded")
while epoch < config.epoch:
    t1 = time.time()
    i = 0
    ptg = 0
    per = len(training_data) // 100
    for sentence, tags in training_data:
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 2. Get our inputs ready for the network, that is, turn them into
        # Tensors of word indices.
        sentence_in = utils.prepare_sequence(sentence, word_to_ix).to(device)
        targets = utils.prepare_sequence(tags, tag_to_ix).to(device)

        # Step 3. Run our forward pass.
        tag_scores = model(sentence_in).to(device)

        # Step 4. Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()
        loss = loss_function(tag_scores, targets)   # only related to current input
        loss.backward()
        optimizer.step()
        if i >= ptg * per:
            print("epoch: {}/{}, {}% completed".format(epoch + 1, config.epoch, ptg))
            ptg += 50
        i += 1
    t2 = time.time()
    epoch += 1
    print("epoch: {}/{}, {} min(s) cost".format(epoch, config.epoch, round((t2 - t1) / 60, 3)))
    state = {}
    state['model'] = model.state_dict()
    state['optimizer'] = optimizer.state_dict()
    state['next_epoch'] = epoch
    torch.save(state, 'snapshot/epoch_{}.pth'.format(epoch))
    torch.save(state, 'snapshot/latest.pth')

print("train completed")