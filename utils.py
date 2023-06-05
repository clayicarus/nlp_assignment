import json
import torch

def read_renmin_raw_dataset():
    # read dataset text like this
    src_path = r"data/renmin/source_BIO_2014_cropus.txt"
    target_path = r"data/renmin/target_BIO_2014_cropus.txt"

    print("reading renmin raw dataset")
    tag_set = open(src_path, "r", encoding = "utf-8")
    t = open(target_path, "r", encoding = "utf-8")
    src_lines = [i for i in tag_set.readlines()]
    target_lines = [i for i in t.readlines()]
    tag_set.close()
    t.close()

    src_list = []
    target_list = []
    for i in src_lines:
        i = i.rstrip()
        if i == '' : 
            continue
        src_list.append(i.split())

    for i in target_lines:
        i = i.rstrip()
        if i == '':
            continue
        target_list.append(i.split())
    
    return src_list, target_list

def read_cluener_raw_dataset(src_path):
    # src_path = r"data/cluener/train.json"
    formal_tag = {
        "organization": ("B_ORG", "I_ORG"), 
        "address": ("B_LOC", "I_LOC"),
        "name": ("B_PER", "I_PER")
    }
    src_list = []
    target_list = []
    f = open(src_path, "r", encoding="utf-8")
    line_num = 0
    for i in f.readlines():
        # print("reading line: {}".format(line_num + 1))
        line_num += 1
        i = i.rstrip()
        if i == '':
            continue
        # print(i)
        j = json.loads(i)
        wd = ' '.join(j["text"]).split()
        target = ['O'] * len(wd)
        label = j["label"]
        # label: {tag1:{embody1, embody2}, tag2}
        for tag in label:
            if tag not in formal_tag:
                continue
            embody = label[tag]
            # embody: {name1: [[poz1], [poz2]], name2: [[]]}
            for name in embody:
                poz_list = embody[name]
                # poz_list: [[0, 1], [2, 2]]
                for poz in poz_list:
                    # poz: [0, 1]
                    target[poz[0]] = formal_tag[tag][0]
                    for idx in range(poz[0] + 1, poz[1] + 1):
                        target[idx] = formal_tag[tag][1]
        assert(len(src_list) == len(target_list))
        src_list.append(wd)
        target_list.append(target)
    f.close()

    return src_list, target_list

def format_cluener_for_glove():
    src_path = r"data/cluener/train.json"
    out_path = r"data/cluener/train.txt"
    f = open(src_path, "r", encoding="utf-8")
    o = open(out_path, "w", encoding="utf-8")
    line_num = 0
    out_li = []
    for i in f.readlines():
        # print("reading line: {}".format(line_num + 1))
        line_num += 1
        i = i.rstrip()
        if i == '':
            continue
        # print(i)
        j = json.loads(i)
        s = ' '.join(j["text"]) + '\n'
        out_li.append(s)
    o.writelines(out_li)
    o.close()
    f.close()

def prepare_sequence(seq, to_ix):
    # change ['The', 'dog'] to [0, 1]
    idxs = [to_ix[w] for w in seq]
    t = torch.tensor(idxs, dtype=torch.long)
    return t