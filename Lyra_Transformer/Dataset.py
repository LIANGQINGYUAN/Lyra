import sys
import os
import json
import pandas as pd
from collections import Counter

import torch
import torch.utils.data as data
from transformers import BertTokenizer,RobertaTokenizer

import re
import parso
from parso.python import tokenize

PAD_TOKEN = '[PAD]'  
UNKNOWN_TOKEN = '[UNK]'  
START_DECODING = '[CLS]'  
STOP_DECODING = '[SEP]' 
CODEBERT_PAD_TOKEN = '<pad>'  
CODEBERT_UNKNOWN_TOKEN = '<unk>'  
CODEBERT_START_DECODING = '<s>'  
CODEBERT_STOP_DECODING = '</s>'  

"""Vocab"""
class Vocab(object):
    def __init__(self, config, mode='encoder-en') -> None:
        super().__init__()
        self.mode = mode
        if self.mode == "encoder-en":
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            self.vocab = self.tokenizer.get_vocab()
        elif self.mode == "encoder-zh":
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
            self.vocab = self.tokenizer.get_vocab()
        elif self.mode == "decoder":
            self.tokenizer = CodeTokenizer(config.vocab_path, bpe_mode=config.bpe_mode)
            self.vocab = self.tokenizer.vocab    

    def size(self):
        return len(self.vocab)


"""Process in pointer generator"""
class PGprocess():
    @classmethod
    def src2ids(self, src_words, src_vocab):
        ids = []
        oovs = []
        unk_id = src_vocab.tokenizer._convert_token_to_id(UNKNOWN_TOKEN)
        for w in src_words:
            i = src_vocab.tokenizer._convert_token_to_id(w)
            if i == unk_id:  # If w is OOV
                if w not in oovs:  # Add to list of OOVs
                    oovs.append(w)
                oov_num = oovs.index(w)  # This is 0 for the first article OOV, 1 for the second article OOV...
                ids.append(src_vocab.size() + oov_num)  # This is e.g. 50000 for the first article OOV, 50001 for the second...
            else:
                ids.append(i)
        return ids, oovs
    @classmethod
    def trg2ids(self, trg_words, tgt_vocab, src_oovs):
        ids = []
        unk_id = tgt_vocab.tokenizer._convert_token_to_id(CODEBERT_UNKNOWN_TOKEN)
        for w in trg_words:
            i = tgt_vocab.tokenizer._convert_token_to_id(w)
            if i == unk_id:  # If w is an OOV word
                if w in src_oovs:  # If w is an in-article OOV
                    vocab_idx = tgt_vocab.size() + src_oovs.index(w)  # Map to its temporary article OOV number
                    ids.append(vocab_idx)
                else:  # If w is an out-of-article OOV
                    ids.append(unk_id)  # Map to the UNK token id
            else:
                ids.append(i)
        return ids
    @classmethod
    def outputids2words(self, id_list, tgt_vocab, src_oovs):
        words = []
        for i in id_list:
            try:
                w = tgt_vocab.tokenizer._convert_id_to_token(i)  # might be [UNK]
            except ValueError as e:  # w is OOV
                assert src_oovs is not None, "Error: model produced a word ID that isn't in the vocabulary. This should not happen in baseline (no pointer-generator) mode"
                src_oov_idx = i - tgt_vocab.size()
                try:
                    w = src_oovs[src_oov_idx]
                    print("_________Copy Worked_________")
                except ValueError as e:  # i doesn't correspond to an article oov
                    raise ValueError(
                        'Error: model produced word ID %i which corresponds to article OOV %i but this example only has %i article OOVs' % (
                        i, src_oov_idx, len(src_oovs)))
            words.append(w)
        return words


"""Dataset of PR in PG Network mode"""
class Code4SQLDataset2PG(data.Dataset):
    def __init__(self, config, dataName='train'):
        self.config = config
        self.encoder_language = config.encoder_language
        self.device = config.device
        self.base_path = config.data_dir
        self.data_file_path = os.path.join(self.base_path, dataName + '.' + config.data_file_suffix)
        self.src_vocab = Vocab(config, self.encoder_language)
        self.trg_vocab = Vocab(config, "decoder")
        self.example_list = self.get_data(self.data_file_path)
        self.length = len(self.example_list)
        self.max_src_oovs = None
        if config.pointer_gen:
            self.max_src_oovs = max([len(ex.enc_oovs) for ex in self.example_list])
            self.src_oovs = [ex.enc_oovs for ex in self.example_list]

    def get_data(self, data_file_path):
        example_list = []
        data_df = pd.read_csv(data_file_path)
        if self.encoder_language == "encoder-en":
            col = "comm_en"
        elif self.encoder_language == "encoder-zh":
            col = "comm_zh"
        for i in range(data_df.shape[0]):
            src_seq = data_df[col][i]
            trg_seq = data_df["code"][i]
            ex = Example(self.config, i, src_seq, trg_seq, self.src_vocab, self.trg_vocab)
            example_list.append(ex)
        return example_list

    def __getitem__(self, index):
        data_index = self.example_list[index].id
        enc_tensor = torch.tensor(self.example_list[index].enc_padded).to(self.device) #as enc input
        dec_input =  torch.tensor(self.example_list[index].dec_input).to(self.device) #as dec input
        dec_target =  torch.tensor(self.example_list[index].dec_target).to(self.device) #as dec target
        assert len(dec_input) == len(dec_target)
        if self.config.pointer_gen:
            enc_wiht_extend_vocab = torch.tensor(self.example_list[index].enc_input_extend_vocab).to(self.device) #as index of scatter_add func
            dec_target =  torch.tensor(self.example_list[index].dec_target).to(self.device) #as ground truth
            assert len(enc_tensor) == len(enc_wiht_extend_vocab)
            return data_index, enc_tensor, enc_wiht_extend_vocab, dec_input, dec_target
        else:
            return data_index, enc_tensor, dec_input, dec_target

    def __len__(self):
        return self.length


class Example(object):
    def __init__(self, config, id, enc, dec, src_vocab, trg_vocab):
        self.id = id  # use id to track each example
        self.max_enc_len = config.max_enc_len
        self.max_dec_len = config.max_dec_len
        # seq to words
        self.enc_words = src_vocab.tokenizer.tokenize(enc)
        enc_ids = src_vocab.tokenizer.encode(self.enc_words)
        self.dec_words = trg_vocab.tokenizer.tokenize(dec)
        dec_ids = trg_vocab.tokenizer.encode(dec)
        # pad the encoder seq
        self.enc_padded = self.init_seq(enc_ids, None, src_vocab, self.max_enc_len)

        # add some info if pg network is allow
        if config.pointer_gen:
            self.enc_input_extend_vocab, self.enc_oovs = PGprocess.src2ids(self.enc_words, src_vocab)
            #pad enc_input_extend_vocab
            self.enc_input_extend_vocab = self.init_seq(self.enc_input_extend_vocab, None, src_vocab, self.max_enc_len)
            #get decode target
            trg_ids_extend_vocab = PGprocess.trg2ids(self.dec_words, trg_vocab, self.enc_oovs)
            self.dec_input, self.dec_target = self.init_seq(dec_ids, trg_ids_extend_vocab, trg_vocab, self.max_dec_len, type="decode")
        else: 
            # pad the decoder seq
            self.dec_input, self.dec_target = self.init_seq(dec_ids, dec_ids, trg_vocab, self.max_dec_len, type="decode")

        #original text
        self.original_src = enc
        if config.bpe_mode:
            self.original_trg = dec
        else:
            self.original_trg = dec.replace("\t","    ")

    def init_seq(self, seq, seq_ext, vocab, max_len, type="encode"):
        if  type == "encode":
            return self.pad_enc_seq(seq, max_len, vocab.tokenizer._convert_token_to_id(PAD_TOKEN))
        elif  type == "decode":
            return self.pad_dec_seq(seq, seq_ext, max_len, vocab.tokenizer._convert_token_to_id(CODEBERT_PAD_TOKEN),  
                                vocab.tokenizer._convert_token_to_id(CODEBERT_START_DECODING), 
                                vocab.tokenizer._convert_token_to_id(CODEBERT_STOP_DECODING))

    def pad_enc_seq(self, seq, max_len, pad_id):
        if len(seq) > max_len:
            seq = seq[:max_len]
        else:
            seq.extend( [pad_id] * (max_len - len(seq)))
        assert len(seq)== max_len
        return seq

    def pad_dec_seq(self, seq, seq_ext, max_len, pad_id, start_decoding, stop_decoding):
        inp = [start_decoding] + seq[:]
        trg = seq_ext[:] + [stop_decoding]
        if len(inp) > max_len:
            inp = inp[:max_len]
            trg = trg[:max_len]
        else:
            inp.extend( [pad_id] * (max_len - len(inp)))
            trg.extend( [pad_id] * (max_len - len(trg)))
        assert len(inp) == len(trg) == max_len 
        return inp, trg


class CodeTokenizer():
    def __init__(self, vocab_path, bpe_mode=False) -> None:
        self.bpe_mode = bpe_mode
        if vocab_path==None:
            pass
        else:
            if bpe_mode:
                self.tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
                self.vocab = self.tokenizer.get_vocab()
            else:
                self.vocab = self.get_vocab_from_file(vocab_path)
                self._word_to_id = self.vocab
                self._id_to_word = {v: k for k, v in self._word_to_id.items()} 

    def _convert_token_to_id(self, word):
        if self.bpe_mode:
            return self.tokenizer._convert_token_to_id(word)
        else:
            if word not in self._word_to_id:
                return self._word_to_id[CODEBERT_UNKNOWN_TOKEN]
            return self._word_to_id[word]

    def _convert_id_to_token(self, word_id):
        if self.bpe_mode:
            return self.tokenizer._convert_id_to_token(word_id)
        else:
            if word_id not in self._id_to_word:
                raise ValueError('Id not found in vocab: %d' % word_id)
            return self._id_to_word[word_id]
        
    def encode(self, s):
        if self.bpe_mode:
            return self.tokenizer.encode(s)[1:-1]
        else:
            return [self._convert_token_to_id(i) for i in self.tokenize(s)] 

    def convert_ids_to_tokens(self, id_list):
        if self.bpe_mode:
            return self.tokenizer.convert_ids_to_tokens(id_list)
        else:
            return [self._convert_id_to_token(i) for i in id_list]

    def convert_tokens_to_string(self, token_list):
        if self.bpe_mode:
            return self.tokenizer.convert_tokens_to_string(token_list)
        else:
            return self.convert_code_tokens_to_string(token_list)

    def convert_code_tokens_to_string(self, token_list):
        return "".join(token_list).replace("😜"," ").replace("<INDENT>","    ")

    def id2sentence(self, word_ids):
        sentence = self.convert_tokens_to_string(self.convert_ids_to_tokens(word_ids)).replace(CODEBERT_PAD_TOKEN,"")
        if CODEBERT_STOP_DECODING in sentence:
            stop_index = sentence.index(CODEBERT_STOP_DECODING)
            s = sentence[:stop_index]
            if len(s)==0 or len(s.replace('.',"")) == '.' :
                return " "
            return s
        else:
            if len(sentence)==0 or len(sentence.replace('.',"")) == 0 :
                return " "
            return sentence

    def tokenize(self, s):
        if self.bpe_mode:
            return self.tokenizer.tokenize(s)
        else:
            s = s.replace("\t", "    ")
            return  self.tokenize_code(s)

    def tokenize_code(self, s):
        token_list = []
        DEFAULT_INDENT = 4
        version_info = parso.utils.parse_version_string("3.8")

        def get_space_list(s):
            space_list = []
            for i in s.split("\n"):
                space_list.append([i.start() for i in re.finditer(' ', i)])
            return space_list

        def add_sapce(word, space_list, l_pos, s_pos):
            # print(word, space_list, l_pos, s_pos)
            if len(word)+s_pos in space_list[l_pos-1]:
                token_list.append(word+"😜")
            else: 
                token_list.append(word)

        def split_underscore(words, space_list, l_pos, s_pos):
            w_list = words.split("_")
            if "_" in words:
                for index in range(len(w_list)):
                    if index<len(w_list)-1:
                        token_list.append(w_list[index])
                        token_list.append("_")
                    else:
                        token_list.append(w_list[index])
                if len(words)+s_pos in space_list[l_pos-1]:
                    token_list[-1] = token_list[-1] + "😜"
            else:
                add_sapce(w_list[0], space_list, l_pos, s_pos)

        code_space_list  = get_space_list(s)
        dict_DENT = {} #line_num : INDENT
        newline_list = []
        for i in tokenize.tokenize(s, version_info):
            if i.type == tokenize.STRING:
                token_list.append(i.string[0])#the symble of '"'
                temp_string = i.string[1:-1]
                space_index=0
                while space_index<len(temp_string) and temp_string[space_index]==" ": 
                    token_list.append("😜") 
                    temp_string = temp_string[space_index+1:]
                space_index=-1
                end_space = 0
                while abs(space_index) <=len(temp_string) and temp_string[space_index]==" ": 
                    # token_list.append("😜") 
                    end_space+=1
                    temp_string = temp_string[:space_index]

                string_space_list = get_space_list(temp_string)
                for j in tokenize.tokenize(temp_string, version_info):
                    if j.type != tokenize.ENDMARKER:
                        # print(j.string)
                        split_underscore(j.string, string_space_list, j.start_pos[0], j.start_pos[1])
                if end_space!=0:
                    [token_list.append("😜") for _ in range(end_space)]
                token_list.append(i.string[-1])#the symble of '"'
                
            elif i.type == tokenize.INDENT or i.type == tokenize.DEDENT:
                s_line, s_pos = i.start_pos
                #recalculate "<INDENT>"
                while len(token_list)!= 0 and token_list[-1] == "<INDENT>":
                    token_list=token_list[:-1]
                
                if s_pos%4 == 0:
                    temp_DENT = []
                    for _ in range(int(s_pos/DEFAULT_INDENT)):
                        token_list.append("<INDENT>")
                        temp_DENT.append("<INDENT>")
                    dict_DENT[s_line] = temp_DENT
                elif s_pos%2 == 0 and s_pos<4:
                    DEFAULT_INDENT=2
                    temp_DENT = []
                    for _ in range(int(s_pos/DEFAULT_INDENT)):
                        token_list.append("<INDENT>")
                        temp_DENT.append("<INDENT>")
                    dict_DENT[s_line] = temp_DENT

            elif i.type == tokenize.NEWLINE:
                # Set the default to keep the last line indented
                token_list.append("\n")
                if len(dict_DENT.keys())!=0:
                    for _ in dict_DENT[list(dict_DENT.keys())[-1]]:
                        token_list.append("<INDENT>")
            
            else:
                split_underscore(i.string, code_space_list, i.start_pos[0], i.start_pos[1])

        if len(token_list) !=0:
            #remove the tail
            while token_list[-1] =='' or token_list[-1] == '<INDENT>':
                token_list = token_list[:-1]
                if len(token_list)==0: return ["😜"]
            return token_list
        else:
            return ["😜"]

    def get_vocab_from_file(self, vocab_file, max_size=50000):
        self._word_to_id = {}
        self._id_to_word = {} 
        self._count = 0  # keeps track of total number of words in the Vocab

        # [UNK], [PAD], [START] and [STOP] get the ids 0,1,2,3.
        for w in [CODEBERT_UNKNOWN_TOKEN, CODEBERT_PAD_TOKEN, CODEBERT_START_DECODING, CODEBERT_STOP_DECODING]:
            self._word_to_id[w] = self._count
            self._id_to_word[self._count] = w
            self._count += 1

        # Read the vocab file and add words up to max_size
        with open(vocab_file, 'r') as vocab_f:
            # the dict of counter
            counter = Counter(json.load(vocab_f))
        for token in [CODEBERT_UNKNOWN_TOKEN, CODEBERT_PAD_TOKEN, CODEBERT_START_DECODING, CODEBERT_STOP_DECODING]:
            del counter[token]

        # alphabetical
        words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
        # list of (word, freq)
        words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

        for w, freq in words_and_frequencies:
            self._word_to_id[w] = self._count
            self._id_to_word[self._count] = w
            self._count += 1
            if max_size > 0 and self._count >= max_size:
                print("max_size of vocab was specified as %i; we now have %i words. Stopping reading." % (max_size, self._count))
                break

        return self._word_to_id    