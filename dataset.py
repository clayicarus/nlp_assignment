from gensim.models.keyedvectors import KeyedVectors

class LSTM_CRF_Dataset:
    # src_list: [["Beijing", "is", "good"], [], ...]
    # target_list: [["B-LOC", "O", "O"], [], ...]
    def __init__(self, src_list, target_list, wv_path = None) -> None:
        self.src_list = src_list
        self.target_list = target_list
        self.training_data = []
        self.word_to_ix = {}
        self.tag_to_ix = {}
        self.word_vectors = []
        self.wv_path = wv_path
        
    def get_dataset(self):
        # dataset: [(["Beijing", "is", "good"], ["B-LOC", "O", "O"]), (...), ...]
        if self.training_data == []:
            for i in range(min(len(self.src_list), len(self.target_list))):
                wd = self.src_list[i]
                lb = self.target_list[i]
                assert(len(wd) == len(lb))
                self.training_data.append((wd, lb))
        return self.training_data
    
    def get_ix_dict(self):
        tr = self.get_dataset()
        if self.word_to_ix == {} or self.tag_to_ix == {}:
            for sent, tag in tr:
                for i in sent:
                    if i not in self.word_to_ix:
                        self.word_to_ix[i] = len(self.word_to_ix)
                for i in tag:
                    if i not in self.tag_to_ix:
                        self.tag_to_ix[i] = len(self.tag_to_ix)
        return self.word_to_ix, self.tag_to_ix
    
    def get_word_vectors(self):
        if self.word_vectors == []:
            res = []
            word_to_ix, _ = self.get_ix_dict()
            assert(self.wv_path != None)
            wv = KeyedVectors.load_word2vec_format(self.wv_path, binary = False, no_header = True)
            assert('<unk>' in wv)
            unk = wv['<unk>']
            for i in word_to_ix:
                if(i not in wv):
                    res.append(unk)
                else:
                    res.append(wv[i])
            self.word_vectors = res
        return self.word_vectors

