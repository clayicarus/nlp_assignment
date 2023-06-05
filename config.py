lr = 0.1  
epoch = 10

# These will usually be more like 32 or 64 dimensional.
# We will keep them small, so we can see how the weights change as we train.
embedding_dim = 100
hidden_dim = 128

cluener_train_set = r"data/cluener/train.json"
cluener_dev_set = r"data/cluener/dev.json"

eval_pth = r"snapshot/latest.pth"
