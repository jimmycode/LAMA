#%%
# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

#%%
#
# Code to generate sentence representations from a pretrained model.
# This can be used to initialize a cross-lingual classifier, for instance.
#

#%%
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import time
import copy
import torch
from torch.nn import functional as F

from src.utils import AttrDict
from src.data.dictionary import Dictionary, BOS_WORD, EOS_WORD, PAD_WORD, UNK_WORD, MASK_WORD
from src.model.transformer import TransformerModel

#%% [markdown]
# ## Reload a pretrained model

#%%
# Perplexity | layers | dropout |
#    9.8509  |   12   |   0.1   | /checkpoint/guismay/dumped/clm_test1/10347724/train.log
#   10.2989  |   18   |   0.1   | /checkpoint/guismay/dumped/clm_test2/10402246/train.log
#   10.7602  |   12   |   0.1   | /checkpoint/guismay/dumped/clm_test3/10431903/train.log
#   11.0479  |   12   |   0.1   | /checkpoint/guismay/dumped/clm_test1/10347726/train.log
#   11.3784  |   12   |   0.1   | /checkpoint/guismay/dumped/clm_test1/10347725/train.log
#   11.8830  |   18   |   0.1   | /checkpoint/guismay/dumped/clm_test2/10403080/train.log
#   12.0149  |   12   |   0.3   | /checkpoint/guismay/dumped/clm_test3/10431904/train.log
#   12.5228  |   18   |   0.1   | /checkpoint/guismay/dumped/clm_test2/10403079/train.log

#%%
# model_path = '/checkpoint/guismay/dumped/clm_test3/10431904/periodic-23.pth'
model_path = '/checkpoint/guismay/dumped/clm_test3/10431904/periodic-23.pth'
reloaded = torch.load(model_path)
params = AttrDict(reloaded['params'])
print("Supported languages: %s" % ", ".join(params.lang2id.keys()))

#%% [markdown]
# ## Build dictionary / update parameters / build model

#%%
# build dictionary / update parameters
dico = Dictionary(reloaded['dico_id2word'], reloaded['dico_word2id'],
                  reloaded['dico_counts'])
assert params.n_words == len(dico)
assert params.bos_index == dico.index(BOS_WORD)
assert params.eos_index == dico.index(EOS_WORD)
assert params.pad_index == dico.index(PAD_WORD)
assert params.unk_index == dico.index(UNK_WORD)
assert params.mask_index == dico.index(MASK_WORD)

# build model / reload weights
model = TransformerModel(params, dico, True, True)
model.load_state_dict(reloaded['model'])
model.cuda()
model.eval()

#%%

#%%
FASTBPE_PATH = '/private/home/guismay/tools/fastBPE/fast'
TOKENIZER_PATH = '/private/home/guismay/tools/mosesdecoder/scripts/tokenizer/tokenizer.perl'
DETOKENIZER_PATH = '/private/home/guismay/tools/mosesdecoder/scripts/tokenizer/detokenizer.perl'
BPE_CODES = '/checkpoint/guismay/ccclean/60000/codes.60000'


#%%
def apply_bpe(txt):
    temp1_path = '/tmp/xxx1'
    temp2_path = '/tmp/xxx2'
    with open(temp1_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(txt) + '\n')
    command = '%s applybpe %s %s %s' % (FASTBPE_PATH, temp2_path, temp1_path,
                                        BPE_CODES)
    os.system(command)
    with open(temp2_path, 'r', encoding='utf-8') as f:
        return [line.rstrip() for line in f]


def tokenize(txt):
    temp1_path = '/tmp/xxx1'
    temp2_path = '/tmp/xxx2'
    with open(temp1_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(txt) + '\n')
    command = 'cat %s | %s -l en -no-escape > %s' % (
        temp1_path, TOKENIZER_PATH, temp2_path)
    os.system(command)
    with open(temp2_path, 'r', encoding='utf-8') as f:
        return [line.rstrip() for line in f]


def detokenize(txt):
    temp1_path = '/tmp/xxx1'
    temp2_path = '/tmp/xxx2'
    with open(temp1_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(txt) + '\n')
    command = 'cat %s | %s -l en > %s' % (temp1_path, DETOKENIZER_PATH,
                                          temp2_path)
    os.system(command)
    with open(temp2_path, 'r', encoding='utf-8') as f:
        return [line.rstrip() for line in f]


#%%
apply_bpe(['Ixtapa ladelopardalis'])

#%%
print(apply_bpe(["Unicorns are awesome"]))
print(detokenize(["This is great don 't you think ?"]))

#%%

#%%
# title = "Vladimir Putin named best FIFA player of the year"
# title = "Pope Francis is getting married"
# title = "Barack Obama shares secrets for perfect abs"
# title = "China sends first panda on the moon"
# title = "Vladimir Putin won the 2019 Ballon d'Or"
# title = "Fake news generator results in unprecedented Wall Street crash"
# title = "Harry Potter lost his magic wand"
# title = "Scientists discovered unicorns living in the Andes Mountains"
# title = "Facebook to increase the salary of PhD students"
# title = "Vladimir Putin outruns Usain Bolt in finals"
# title = "Bill Gates goes on vacation with Taylor Swift"
# title = "Convolutional neural networks are not working"
# title = "Bachar el-Assad meets Harry Potter"
# title = "Scientists discovered unicorns living in the Andes Mountains"
# title = "Elon Musk sells drugs to teenagers"

# title = "%s TITLE2ARTICLE" % title
# title = "Facebook to increase the salary of"
# title = "Peter F. Martin is born in"
title = "Tan Jiexi is born in"

title = [title]
title = tokenize(title)
title = apply_bpe(title)
title = "</s> %s" % title[0]
# title = "</s> %s TL@@ ;@@ DR" % title[0]
print(title)


def sample(scores, temperature, topk):
    if topk > 0:
        mask = scores < scores.topk(topk, dim=1, largest=True,
                                    sorted=True)[0][:, -1:]
        scores[mask] = -1e9
    return torch.multinomial(F.softmax(scores / temperature, dim=1),
                             1).squeeze(1)


DEVICE = 'cuda:0'
max_len = 70
temperature, topk = 0.7, 40
# temperature, topk = None, 5
# assert (temperature is None) == (topk is None)
n = 10

#%%
# create batch
init_len = len(title.split())
init_lengths = torch.LongTensor(n).fill_(init_len).to(DEVICE)
word_ids = torch.LongTensor(init_len + max_len,
                            n).fill_(params.pad_index).to(DEVICE)
word_ids[:init_len] = torch.LongTensor([dico.index(w) for w in title.split()
                                        ]).to(DEVICE)[:, None]

cache = {'slen': 0}
start = time.time()

with torch.no_grad():
    for cur_len in range(max_len):
        # scores for next words
        tensor = model('fwd',
                       x=word_ids[:init_len + cur_len],
                       lengths=init_lengths + cur_len,
                       langs=None,
                       causal=True,
                       cache=cache).contiguous()
        scores = model.pred_layer.proj(tensor[-1])

        # sample next word
        if temperature is None and topk is None:
            next_words = torch.topk(scores, 1)[1].squeeze(1)
        else:
            next_words = sample(scores, temperature, topk)

        # update batch
        word_ids[
            init_len +
            cur_len] = next_words if cur_len < max_len - 1 else params.eos_index

        # early stopping
        if ((word_ids == params.eos_index).sum(0) < 2).sum().item() == 0:
            break

tok = []
for i in range(n):
    wid = [dico[word_ids[j, i].item()] for j in range(len(word_ids))][1:]
    wid = wid[:wid.index(EOS_WORD)] if EOS_WORD in wid else wid
    tok.append(" ".join(wid).replace("@@ ", ""))
detok = detokenize(tok)

outputs = []
for i, s in enumerate(detok):
    # s = s[s.find('TITLE2ARTICLE') + len('TITLE2ARTICLE'):].strip()
    s = s.replace("TITLE2ARTICLE", "\n\n")
    s = s.replace("NEWLINE", "\n")
    s = "\n".join([s.strip() for s in s.split("\n")])
    outputs.append(s)

for o in outputs:
    print("=" * 20)
    print("")
    print(o)
    print("")
    print(time.time() - start)

#%%