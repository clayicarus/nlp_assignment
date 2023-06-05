import os
import torch
import utils
import config
import torch.nn as nn
import numpy as np
from model.lstm_crf import *
from dataset import LSTM_CRF_Dataset
from torch import optim

a, b = utils.read_cluener_raw_dataset(config.cluener_train_set)
dataset = LSTM_CRF_Dataset(a, b)
dev_data = dataset.get_dataset()
word_to_ix, tag_to_ix = dataset.get_ix_dict()

device = torch.device("cpu")

model = LSTM_CRF(config.embedding_dim, config.hidden_dim, len(word_to_ix), len(tag_to_ix)).to(device)
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr = config.lr)

assert os.path.exists(config.eval_pth)

state = torch.load(config.eval_pth)
model.load_state_dict(state['model'])
optimizer.load_state_dict(state['optimizer'])
print("{} loaded".format(config.eval_pth))

def predict(word_list: list):
    predict = []
    with torch.no_grad():
        inputs = utils.prepare_sequence(word_list, word_to_ix).to(device)
        tag_scores = model(inputs).to(device)
        # print(tag_scores)
        scores_mat = tag_scores.to("cpu").detach().numpy()
        tag_list = [i for i in tag_to_ix]
        predict = [tag_list[np.argmax(i)] for i in scores_mat]
    return predict

def eval_dev():
    acc = 0
    for src, tag in dev_data:
        pred = predict(src)
        tot = min(len(pred), len(tag))
        ac = 0
        for i in range(tot):
            if pred[i] == tag[i]:
                ac += 1
        ac = ac / tot
        acc += ac
    acc /= len(dev_data)
    print("acc: {}".format(round(acc, 3)))


if __name__ == "__main__":
    # eval_dev()
    s = input()
    while s != '':
        l = ' '.join(s).split()
        pred = predict(l)
        print(l)
        print(pred)
        s = input()