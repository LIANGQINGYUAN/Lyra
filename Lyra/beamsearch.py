import torch
import torch.nn as nn
import torch.nn.functional as F
from Model import Model
import Dataset
from Dataset import Vocab, PGprocess

import os
import sys
import time
import json
from tqdm import tqdm
import pandas as pd

from utils import *

def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx)

def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''
    sz_b, len_s = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask

class BeamSearch(nn.Module):
    ''' Load a trained model and generate in beam search fashion. '''
    def __init__(
            self, model, params, beam_size, max_seq_len,
            src_pad_idx, trg_pad_idx, trg_bos_idx, trg_eos_idx):
        
        super(BeamSearch, self).__init__()

        self.alpha = 0.7
        self.params = params
        self.beam_size = beam_size
        self.max_seq_len = max_seq_len
        self.src_pad_idx = src_pad_idx
        self.trg_bos_idx = trg_bos_idx
        self.trg_eos_idx = trg_eos_idx

        self.model = model
        self.model.eval()
        #init_seq: [1,1] : [[bos_idx]]
        self.register_buffer('init_seq', torch.LongTensor([[trg_bos_idx]]))
        #blank_seqs: [beam_size, max_seq_len], [bos_idx, pad_idx, pad_idx, ......]
        self.register_buffer(
            'blank_seqs', 
            torch.full((beam_size, max_seq_len), trg_pad_idx, dtype=torch.long))
        self.blank_seqs[:, 0] = self.trg_bos_idx
        #len_map: [1, max_seq_len] : [1, 2, ... , max_seq_len]
        self.register_buffer(
            'len_map', 
            torch.arange(1, max_seq_len + 1, dtype=torch.long).unsqueeze(0))

    '''
    模型解码
    '''
    def _model_decode(self, trg_seq, enc_output, src_mask):
        # trg_seq: [beam_size, step]
        # trg_mask: [beam_size, step, step]
        # enc_output: [beam_size, seq_len, d_model]
        trg_mask = get_subsequent_mask(trg_seq)
        # dec_output: [beam_size, step, d_model]
        preb = self.model.decode(trg_seq, trg_mask, enc_output, src_mask, self.enc_wiht_extend_vocab, self.max_ext_len)
        return preb
        #return self.model.trg_word_prj(dec_output)*(enc_output.size(2) ** -0.5)

    '''
    获取初始化状态
    '''
    def _get_init_state(self, src_seq, src_mask):
        #src_seq: [1, seq_len]
        beam_size = self.beam_size
        #enc_output: [1, seq_len, d_model]
        enc_output = self.model.encode(src_seq, src_mask)
        #dec_output: [1, 1, vocab_size]
        dec_output = self._model_decode(self.init_seq, enc_output, src_mask)#use self.init_seq
        #best_k: [[1, beam_size]]
        best_k_probs, best_k_idx = dec_output[:, -1, :].topk(beam_size)
        #print(">>>>>>>>>src_seq: ", src_seq)
        #print(">>>>>>>>>best_k_probs: ",best_k_probs)
        #print(">>>>>>>>>best_k_idx: ",best_k_idx)

        #socres: [beam_size]
        scores = torch.log(best_k_probs).view(beam_size)
        #gen_seq: [beam_size, max_seq_len]
        gen_seq = self.blank_seqs.clone().detach()
        gen_seq_ext = self.blank_seqs.clone().detach()
        #[ [bos_id, pred_r0_1, pad, pad, ... ],
        #  [bos_id, pred_r1_1, pad, pad, ... ],
        #  [bos_id, pred_r2_1, pad, pad, ... ],
        #  [bos_id, pred_r3_1, pad, pad, ... ] ]
        best_k_idx_list = []
        #print("best_k_idx: ", best_k_idx)
        for i in best_k_idx[0]:
            if i>=self.params.dec_vocab_size:
                best_k_idx_list.append(0)
            else:
                best_k_idx_list.append(i)
        best_k_idx_no_oovs = torch.LongTensor(best_k_idx_list)
        gen_seq[:, 1] = best_k_idx_no_oovs
        gen_seq_ext[:, 1] = best_k_idx[0]
        #enc_output: [beam_size, seq_len, d_model]
        enc_output = enc_output.repeat(beam_size, 1, 1)
        return enc_output, gen_seq, gen_seq_ext, scores


    def _get_the_best_score_and_idx(self, gen_seq, gen_seq_ext, dec_output, scores, step):
        #gen_seq: [beam_size, max_seq_len]
        #dec_output: [beam_size, step, vocab_size]
        assert len(scores.size()) == 1
        beam_size = self.beam_size
        # Get k candidates for each beam, k^2 candidates in total.
        # dec_output[:, -1, :]: [beam_size, vocab_size], -1 indicate the last step
        # best_k2_probs,best_k2_idx : [beam_size,beam_size]
        best_k2_probs, best_k2_idx = dec_output[:, -1, :].topk(beam_size)
        # Include the previous scores.
        scores = torch.log(best_k2_probs).view(beam_size, -1) + scores.view(beam_size, 1)
        if torch.isnan(scores).any():
            print("Error: log probs contains NAN!")
        # Get the best k candidates from k^2 candidates.
        # best_k_idx_in_k2: [4]
        scores, best_k_idx_in_k2 = scores.view(-1).topk(beam_size)
        # Get the corresponding positions of the best k candidiates.
        best_k_r_idxs, best_k_c_idxs = best_k_idx_in_k2 // beam_size, best_k_idx_in_k2 % beam_size
        best_k_idx = best_k2_idx[best_k_r_idxs, best_k_c_idxs]
        # Copy the corresponding previous tokens.
        gen_seq[:, :step] = gen_seq[best_k_r_idxs, :step]
        gen_seq_ext[:, :step] = gen_seq_ext[best_k_r_idxs, :step]
        # Set the best tokens in this beam search step
        best_k_idx_list = []
        for i in best_k_idx:
            if i>=self.params.dec_vocab_size:
                best_k_idx_list.append(0)
            else:
                best_k_idx_list.append(i)
        best_k_idx_no_oovs = torch.LongTensor(best_k_idx_list)
        gen_seq[:, step] = best_k_idx_no_oovs
        gen_seq_ext[:, step]  = best_k_idx

        return gen_seq, gen_seq_ext, scores


    def generate_sentence(self, src_seq, enc_wiht_extend_vocab, max_ext_len):
        # Only accept batch size equals to 1 in this function.
        # TODO: expand to batch operation.
        assert src_seq.size(0) == 1
        # set params
        src_pad_idx, trg_eos_idx = self.src_pad_idx, self.trg_eos_idx 
        max_seq_len, beam_size, alpha = self.max_seq_len, self.beam_size, self.alpha 
        self.enc_wiht_extend_vocab = enc_wiht_extend_vocab
        self.max_ext_len = max_ext_len
        with torch.no_grad():
            #get mask of src_seq
            src_mask = get_pad_mask(src_seq, src_pad_idx)
            #get init enc_output, gen_seq and scores:
            # - enc_output: [beam_size, seq_len, d_model]
            # - gen_seq: [beam_size, seq_len]
            # - scores: [beam_size]
            enc_output, gen_seq, gen_seq_ext, scores = self._get_init_state(src_seq, src_mask)

            ans_idx = 0   # default
            for step in range(2, max_seq_len):    # decode up to max length
                # dec_output: [beam_size, step, vocab_size]
                dec_output = self._model_decode(gen_seq[:, :step], enc_output, src_mask)
                # gen_seq: [beam_size, max_seq_len]
                # scores: [beam_size]
                gen_seq, gen_seq_ext, scores = self._get_the_best_score_and_idx(gen_seq, gen_seq_ext, dec_output, scores, step)

                # Check if all path finished
                # -- locate the eos in the generated sequences
                # -- eos_locs size: [beam_size, max_seq_len], True of False
                eos_locs = gen_seq_ext == trg_eos_idx  
                # -- replace the eos with its position for the length penalty use
                # -- seq_lens size: [beam_size]
                # 对eos_locs取反，然后将为True的地方填充数值为max_seq_len，最终原来eos_locs为True的保持不变
                # self.len_map.size(): [1, max_seq_len]
                # self.len_map.masked_fill(~eos_locs, max_seq_len).size(): [beam_size, max_seq_len]
                seq_lens, _ = self.len_map.masked_fill(~eos_locs, max_seq_len).min(1)
                # -- check if all beams contain eos
                if (eos_locs.sum(1) > 0).sum(0).item() == beam_size:
                    # TODO: Try different terminate conditions.
                    # 每个分数处以序列长度的alpha次方，选出得分最大的beam
                    _, ans_idx = scores.div(seq_lens.float() ** alpha).max(0)
                    ans_idx = ans_idx.item()
                    break
        return gen_seq_ext[ans_idx][:seq_lens[ans_idx]].tolist()


class Search(object):
    def __init__(self, params, model_file_path, dataloader, data_file_prefix="test"):
        self.model_file_path = model_file_path
        # param
        self.params = params
        self.data_file_prefix = data_file_prefix
        self.vocab = Vocab(params, mode="decoder")
        self.dataloader = dataloader
        self.bos_id=self.vocab.tokenizer._convert_token_to_id(Dataset.CODEBERT_START_DECODING)
        self.eos_id=self.vocab.tokenizer._convert_token_to_id(Dataset.CODEBERT_STOP_DECODING)
        
        # model
        checkpoint = torch.load(model_file_path)
        model_opt = checkpoint['settings']
        self.model = Model(params).to(self.params.device)   
        self.model.load_state_dict(checkpoint['model'])
        print('[Info] Trained model state loaded.')

        # generator
        self.generator = BeamSearch(
            model=self.model,
            params=params,
            beam_size=self.params.beam_size,
            max_seq_len=self.params.max_dec_len,
            src_pad_idx=0,
            trg_pad_idx=1,
            trg_bos_idx=self.bos_id,
            trg_eos_idx=self.eos_id).to(self.params.device)

    def get_evaluation(self, hyps_list, gold_list):
        result_dict = get_rouge_dict(hyps_list, gold_list)
        result_dict['bleu'] = get_bleu4_score(hyps_list, gold_list, self.vocab.tokenizer)
        result_dict['excutable_rate'] = get_executable_rate(hyps_list)
        rate, index = get_func_correctness(hyps_list, gold_list, repalce_string=True, need_index=True)
        result_dict['AST_same_without_str_rate'] = rate
        result_dict['exact_without_str_index'] = index
        rate , index  = get_func_correctness(hyps_list, gold_list, need_index=True)
        result_dict['func_correctness_rate'] = rate
        result_dict['exact_index'] = index
        return result_dict

    def store_res(self, hyps_list, refs_list):
        # rouge and bleu
        result_dict = self.get_evaluation(hyps_list, refs_list)
        # store the decode result
        sotre_res = pd.DataFrame()
        sotre_res['hyps'] = hyps_list
        sotre_res['refs'] = refs_list
        i = self.model_file_path.rindex('model/')
        j = self.model_file_path.rindex('.')
        decode_res_name = self.model_file_path[i+6:j] + '_decode.csv'
        decode_path = self.model_file_path[:i] + "decode"
        if not os.path.exists(decode_path):
            os.makedirs(decode_path)
        sotre_res.to_csv(os.path.join(decode_path,decode_res_name))
        # store the performance
        with open(os.path.join(decode_path, self.model_file_path[i+6:j] + "_result_dict.txt"), 'w') as f:
            json.dump(result_dict, f, indent=4)
        return result_dict

    def decode(self):
        device = torch.device(self.params.device)
        hyps_list = []
        refs_list = []
        desc = '  - (Testing) '
        for batch in tqdm(self.dataloader,  mininterval=2, desc=desc):
            if self.params.pointer_gen:
                data_index, enc_batch, enc_wiht_extend_vocab, dce_input, dce_target = batch
                max_ext_len = self.dataloader.dataset.max_src_oovs
                src_oovs = [self.dataloader.dataset.src_oovs[i] for i in data_index][0]
            else:
                src_oovs,enc_wiht_extend_vocab,max_ext_len = None,None,None
                data_index, enc_batch, dce_input, dce_target = batch
            # print("input data: ", enc_batch, dce_input, dce_target)
            pred_seq = self.generator.generate_sentence(enc_batch, enc_wiht_extend_vocab, max_ext_len)
            # print("pred_seq: ", pred_seq)
            if self.params.pointer_gen:
                pred_line = self.vocab.convert_tokens_to_string(PGprocess.outputids2words(pred_seq, self.vocab, src_oovs)).replace(Dataset.CODEBERT_START_DECODING,"").replace(Dataset.CODEBERT_PAD_TOKEN,"")
            else:
                pred_line = self.vocab.tokenizer.id2sentence(pred_seq).replace(Dataset.CODEBERT_START_DECODING,"")
            ref_line = self.dataloader.dataset.example_list[data_index].original_trg
            ecn_line = self.dataloader.dataset.example_list[data_index].original_src
            print()
            print(">>>>>>data_index: ",data_index)
            print(">>>>>>ref_line: ",ref_line)
            print(">>>>>>ecn_line: \n",ecn_line)
            print(">>>>>>pred_seq: ",pred_seq)
            print(">>>>>>pred_line: \n",pred_line)
            print()
            
            hyps_list.append(self.vocab.tokenizer.convert_tokens_to_string(self.vocab.tokenizer.tokenize(pred_line)))
            refs_list.append(self.vocab.tokenizer.convert_tokens_to_string(self.vocab.tokenizer.tokenize(ref_line)))
        result_dict=self.store_res(hyps_list, refs_list)
        return result_dict

# transformer
# python -m run decode models/train_1618824548/model
# python -m run decode models/train_1618824588/model

# python -m run decode models/train_1618824614/model
# python -m run decode models/train_1618824645/model

# 
# python -m run decode models/train_1619665828/model
# python -m run decode models/train_1619665898/model