# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from .xlm.utils import AttrDict
from .xlm.data.dictionary import Dictionary, BOS_WORD, EOS_WORD, PAD_WORD, UNK_WORD, MASK_WORD
from .xlm.model.transformer import TransformerModel

import numpy as np
from typing import List
from lama.modules.base_connector import *

FASTBPE_PATH = '/private/home/guismay/tools/fastBPE/fast'
TOKENIZER_PATH = '/private/home/guismay/tools/mosesdecoder/scripts/tokenizer/tokenizer.perl'
DETOKENIZER_PATH = '/private/home/guismay/tools/mosesdecoder/scripts/tokenizer/detokenizer.perl'
BPE_CODES = '/checkpoint/guismay/ccclean/60000/codes.60000'


def apply_bpe(txt):
    temp1_path = '/tmp/xxx1'
    temp2_path = '/tmp/xxx2'
    with open(temp1_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(txt) + '\n')
    command = '%s applybpe %s %s %s' % (FASTBPE_PATH, temp2_path, temp1_path, BPE_CODES)
    os.system(command)
    with open(temp2_path, 'r', encoding='utf-8') as f:
        return [line.rstrip() for line in f]


def tokenize(txt):
    temp1_path = '/tmp/xxx1'
    temp2_path = '/tmp/xxx2'
    with open(temp1_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(txt) + '\n')
    command = 'cat %s | %s -l en -no-escape > %s' % (temp1_path, TOKENIZER_PATH, temp2_path)
    os.system(command)
    with open(temp2_path, 'r', encoding='utf-8') as f:
        return [line.rstrip() for line in f]


def detokenize(txt):
    temp1_path = '/tmp/xxx1'
    temp2_path = '/tmp/xxx2'
    with open(temp1_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(txt) + '\n')
    command = 'cat %s | %s -l en > %s' % (temp1_path, DETOKENIZER_PATH, temp2_path)
    os.system(command)
    with open(temp2_path, 'r', encoding='utf-8') as f:
        return [line.rstrip() for line in f]


class XLM(Base_Connector):

    def __init__(self, args):
        super().__init__()

        if args.xlm_model_path is not None:
            # load bert model from file
            model_path = args.xlm_model_path
            print("loading XLM model from {}".format(model_path))
        else:
            raise ValueError("Model path missing.")

        reloaded = torch.load(model_path)
        params = AttrDict(reloaded['params'])
        print("[XLM] Supported languages: %s" % ", ".join(params.lang2id.keys()))

        if params.asm == True:
            params.asm = False
            print("Warning: params.asm flag is True, changed to False.")

        # build dictionary / update parameters
        self.dico = Dictionary(reloaded['dico_id2word'], reloaded['dico_word2id'],
                               reloaded['dico_counts'])
        assert params.n_words == len(self.dico)
        assert params.bos_index == self.dico.index(BOS_WORD)
        assert params.eos_index == self.dico.index(EOS_WORD)
        assert params.pad_index == self.dico.index(PAD_WORD)
        assert params.unk_index == self.dico.index(UNK_WORD)
        assert params.mask_index == self.dico.index(MASK_WORD)
        print(params)

        # build model / reload weights
        self.xlm_model = TransformerModel(params, self.dico, True, True)
        self.xlm_model.load_state_dict(reloaded['model'])
        # model.cuda()
        self.xlm_model.eval()

        # # GPT uses different way to represent BPE then BERT. Namely, the
        # # final suffixes are indicated with </w> suffix, while pieces that must
        # # be followed are written as is. In BERT the prefixes are written as is
        # # while the parts that must follow (not be followed!) have '##' prefix.
        # # There is no one-to-one coversion. But at least we may make pieces that
        # # may form a full word look the same.
        # # Note that we should be very careful now,
        # # tokenizer.convert_tokens_to_ids won't work with our vocabulary.
        # def convert_word(word):
        #     if word == OPENAI_UNK:
        #         return word
        #     if word == '\n</w>':
        #         # Redefine symbol EOS to improve visualization.
        #         return OPENAI_EOS
        #     return word[:-4] if word.endswith('</w>') else f'{word}##'

        # _, gpt_vocab = zip(*sorted(self.tokenizer.decoder.items()))
        # self.vocab = [convert_word(word) for word in gpt_vocab]

        self.vocab = [self.dico[i] for i in range(len(self.dico))]
        self._init_inverse_vocab()

        # Get UNK symbol as it's written in the origin XLM vocab.
        self.unk_symbol = UNK_WORD
        self.eos_id = self.inverse_vocab[EOS_WORD]
        self.model_vocab = self.vocab

    def _init_inverse_vocab(self):
        self.inverse_vocab = {w: self.dico.index(w) for w in self.vocab}

    def _cuda(self):
        self.xlm_model.cuda()

    def get_id(self, string):
        tokenized_text = self._tokenize(string)
        indexed_tokens = [self.dico.index(w) for w in tokenized_text.split()]

        # tokenized_text = self.tokenizer.tokenize(string)
        # indexed_string = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        # indexed_string = self.convert_ids(indexed_string)
        return indexed_tokens

    def _tokenize(self, text: str) -> List[str]:
        tokenized_text = tokenize([text])
        bpe_text = apply_bpe(tokenized_text)[0]
        return bpe_text.split()

    def __get_input_tensors(self, sentence_list):
        """Concatenates, tokenize and converts a sentences to model inputs.

        Args:
            sentence_list: A list of strings. The string may contain a special
            [MASK] token.

        Returns:
            A tuple (src_tensor, dst_tensor, masked_indices, tokenized_text).
                src_tensor: torch.LongTensor with shape (seq_len), the input to
                    the new without the last symbol and with EOS prepended.
                dst_tensor: torch.LongTensor with shape (seq_len).
                masked_indices: A list of indices of [MASK] in dst_tensor.
                seq_length: length of the sequence
        """
        # Split the sentence by [MASK] and tokenize the chunks independently.
        tokenized_text = []
        masked_indices = []
        for sentence in sentence_list:
            tokenized_text.append(EOS_WORD)  # prepend </s> symbol

            sentence = sentence.replace('[MASK]', self.unk_symbol)
            cur_tokens = self._tokenize(sentence)
            mask_index = cur_tokens.index(
                self.unk_symbol) + len(tokenized_text) - 1  # left-shift for uni-directional LM
            assert mask_index >= 0
            masked_indices.append(mask_index)
            tokenized_text += cur_tokens

        # full_indexed_tokens = [ self.eos_id ] + self.tokenizer.convert_tokens_to_ids(tokenized_text)
        full_indexed_tokens = [self.dico.index(w) for w in tokenized_text]

        full_tokens_tensor = torch.tensor(full_indexed_tokens)
        src_tensor = full_tokens_tensor[:-1]
        dst_tensor = full_tokens_tensor[1:]
        seq_length = src_tensor.size(0)

        return src_tensor, dst_tensor, masked_indices, seq_length

    def get_batch_generation(self, sentences_list, logger=None, try_cuda=True):
        if try_cuda:
            self.try_cuda()

        src_tensor_list, dst_tensor_list, masked_indices_list, seq_lens = zip(
            *[self.__get_input_tensors(sentences) for sentences in sentences_list])

        src_tensor_batch = torch.nn.utils.rnn.pad_sequence(src_tensor_list,
                                                           batch_first=True,
                                                           padding_value=self.dico.index(PAD_WORD))
        src_tensor_batch = src_tensor_batch.transpose_(0, 1)  # [slen, bs]
        seq_lens_tensor = torch.LongTensor(seq_lens)

        # The model uses shared embedding space for tokens and positions. More
        # precisely, the first len(vocab) indidices are reseved for words, the
        # last n_special symbols are reserved for special symbols and the rest
        # is used for positions. Softmax and embedding matrices are shared and
        # as result some of output "symbols" correspond to positions. To fix
        # that we have to manually remove logits for positions.
        with torch.no_grad():
            hidden = self.xlm_model('fwd',
                                    x=src_tensor_batch.to(self._model_device),
                                    lengths=seq_lens_tensor.to(self._model_device),
                                    langs=None,
                                    causal=True,
                                    cache={'slen': 0})
            assert hidden.dim() == 3
            hidden_sizes = hidden.size()

            logits = self.xlm_model.pred_layer.get_scores(hidden.view(-1, hidden_sizes[-1]))
            logits = logits.view(hidden_sizes[0], hidden_sizes[1],
                                 -1).contiguous()  # [slen, bs, vocab]
            logits = logits.transpose_(0, 1)  # [bs, slen, vocab]

            # logits = logits[..., :self.gpt_model.config.vocab_size]
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1).cpu()

        token_ids_list = [np.array(dst_tensor.numpy()) for dst_tensor in dst_tensor_list]

        return log_probs, token_ids_list, masked_indices_list

    # def get_contextual_embeddings(self, sentences_list, try_cuda=True):

    #     if try_cuda:
    #         self.try_cuda()

    #     src_tensor_list, dst_tensor_list, masked_indices_list, _ = zip(
    #         *[self.__get_input_tensors(sentences) for sentences in sentences_list])

    #     src_tensor_batch = torch.nn.utils.rnn.pad_sequence(src_tensor_list, batch_first=True)

    #     with torch.no_grad():
    #         output = self.gpt_model.transformer(src_tensor_batch.to(self._model_device))

    #     # TODO
    #     sentence_lengths = None
    #     tokenized_text_list = None

    #     # As we only return the last layer, [] to have the same format as other models
    #     return [output], sentence_lengths, tokenized_text_list
