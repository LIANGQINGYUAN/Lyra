# -*- coding: utf-8 -*-
#base
import os
os.chdir("./")
import fire
import time
import json
from tqdm import tqdm
#torch
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
#module
from ScheduledOptim import ScheduledOptim
from Dataset import Vocab, Code4SQLDataset2PG, PGprocess
from Model import Model
from beamsearch import Search
from utils import *

# logging
import wandb
wandb.init(project="code4sql")

__author__ = "Qingyuan Liang"

"""Params"""
class Params():
    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)
    
    def add(self, key, value):
        self.__dict__[key] = value

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__


"""Base class of all process-related classes in order to share similar process"""
class Procedure(object):
    def __init__(self, params):
        # data
        self.src_vocab = Vocab(params, mode=params.encoder_language)
        self.trg_vocab = Vocab(params, mode="decoder")
        params.add("enc_vocab_size", self.src_vocab.size())
        params.add("dec_vocab_size", self.trg_vocab.size())
        wandb.config.update(params.dict)
        self.params = params

    def get_model_dir(self):
        #save path
        cur_time = time.time()
        train_dir = os.path.join(self.params.model_root, 'train_%d' % (cur_time))
        if not os.path.exists(train_dir):
            os.makedirs(train_dir)
        self.model_dir = os.path.join(train_dir, 'model')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        param_path = os.path.join(self.model_dir, 'params.json')
        print("Dump hyper-parameters to {}.".format(param_path))
        self.params.save(param_path)
        return self.model_dir

    def get_save_path(self, iter):
        cur_time = time.time()
        prefix = 'model_{}_{}'
        param_prefix = 'params_{}_{}'
        model_save_path = os.path.join(self.model_dir, prefix.format(iter, cur_time))
        param_save_path = os.path.join(self.model_dir, param_prefix.format(iter, cur_time))
        return model_save_path, param_save_path

    def setup_data(self, mode='train'):
        if mode == 'train':
            dataset = Code4SQLDataset2PG(self.params, 'train')
            dataloader = DataLoader(dataset, self.params.batch_size, shuffle=True, drop_last=True)
        elif mode == 'valid':
            dataset = Code4SQLDataset2PG(self.params, 'valid')
            dataloader = DataLoader(dataset, self.params.batch_size, shuffle=False, drop_last=True)
        else:
            dataset = Code4SQLDataset2PG(self.params, 'test')
            dataloader = DataLoader(dataset, batch_size = 1, shuffle=False)
        return dataloader

    def setup_train(self):
        # model
        model = Model(self.params)
        # parallel
        #model = torch.nn.DataParallel(model, device_ids=[5, 1, 2]).to(self.params.device)
        model = model.to(self.params.device)
        wandb.watch(model)
        # optim
        # optimizer = optim.Adam(model.parameters(), lr=self.params.lr)
        #optimizer = optim.Adagrad(model.parameters(), lr=self.params.lr, initial_accumulator_value=0.1)
        optimizer = ScheduledOptim(
            optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-09, lr=self.params.lr),
            0.1, self.params.d_model, self.params.n_warmup_steps)
        return model, optimizer

    def train_epoch(self, model, training_data, max_ext_len, optimizer):
        model.train()
        total_loss, total_rouge, total_bleu , n_batch = 0, 0, 0, 0
        total_func_corr = 0
        desc = '  - (Training)   '
        for batch in tqdm(training_data, mininterval=2, desc=desc):
            if self.params.pointer_gen:
                data_index, enc_batch, enc_wiht_extend_vocab, dec_input, dec_target = batch
                src_oovs = [training_data.dataset.src_oovs[i] for i in data_index]
            else:
                src_oovs,enc_wiht_extend_vocab,max_ext_len = None,None,None
                data_index, enc_batch, dec_input, dec_target = batch

            # forward
            optimizer.zero_grad()
            pred, loss = model(enc_batch, dec_input, dec_target, enc_wiht_extend_vocab, max_ext_len)
            #optim
            loss.backward()
            optimizer.step()
            #rouge, bleu
            original_trg_batch = [training_data.dataset.example_list[i].original_trg for i in data_index]
            r_score, b_score, func_corr = self.cal_bleu_rouge(pred, original_trg_batch, src_oovs)
            #sum and averge
            total_loss += loss.item()
            total_rouge += r_score
            total_bleu += b_score
            total_func_corr += func_corr
            n_batch += 1

        return total_loss/n_batch, total_rouge/n_batch, total_bleu/n_batch, total_func_corr/n_batch

    def eval_epoch(self, model, validation_data, max_ext_len):
        model.eval()
        total_loss, total_rouge, total_bleu , n_batch = 0, 0 , 0, 0
        total_func_corr = 0
        desc = '  - (Validation) '
        with torch.no_grad():
            for batch in tqdm(validation_data, mininterval=2, desc=desc):
                if self.params.pointer_gen:
                    data_index, enc_batch, enc_wiht_extend_vocab, dec_input, dec_target = batch
                    src_oovs = [validation_data.dataset.src_oovs[i] for i in data_index]
                else:
                    src_oovs,enc_wiht_extend_vocab,max_ext_len = None,None,None
                    data_index, enc_batch, dec_input, dec_target = batch

                # forward
                pred, loss = model(enc_batch, dec_input, dec_target, enc_wiht_extend_vocab, max_ext_len)
                #print("xxxxxxxxxxx",loss)
                #rouge, bleu
                original_trg_batch = [validation_data.dataset.example_list[i].original_trg for i in data_index]
                r_score, b_score, func_corr = self.cal_bleu_rouge(pred, original_trg_batch, src_oovs)
                #sum and averge
                total_loss += loss.item()
                total_rouge += r_score
                total_bleu += b_score
                total_func_corr += func_corr
                n_batch += 1

            return total_loss/n_batch, total_rouge/n_batch, total_bleu/n_batch, total_func_corr/n_batch

    def cal_bleu_rouge(self, pred, original_trg_batch, src_oovs=None):
        # print("pred: \n", pred)
        # print("pred: \n", pred.shape)
        pred_batch = torch.max(pred, dim=-1)[1]
        # print("pred_batch: \n", pred_batch)
        pred_sentences = []
        gold_sentences = original_trg_batch
        for i in range(pred_batch.shape[0]):
            if self.params.pointer_gen:
                pred_sentences.append(self.trg_vocab.tokenizer.convert_tokens_to_string(PGprocess.outputids2words(pred_batch.cpu().numpy().tolist()[i], self.vocab, src_oovs[i])))
            else:
                pred_sentences.append(self.trg_vocab.tokenizer.id2sentence(pred_batch.cpu().numpy().tolist()[i]))
        
        print()
        print(">>>>>>>>>>>Pred_sentences example: \n", pred_sentences[0])
        print()
        print(">>>>>>>>>>>Gold_sentences example: \n", gold_sentences[0])
        print()
        
        #rouge l
        r_score = get_rouge_dict(pred_sentences, gold_sentences)["rouge-l"]['f']
        #bleu 4
        b_score = get_bleu4_score(pred_sentences, gold_sentences, self.trg_vocab.tokenizer)
        #func_correctness
        func_corr = get_func_correctness(pred_sentences, gold_sentences)
                  
        return r_score, b_score, func_corr


'''Main class'''
class Main():
    def __init__(self, param_path=None):
        if param_path != None:
            self.params = Params(param_path)
            self.base = Procedure(self.params)

    def train(self):
        self.model_dir = self.base.get_model_dir()
        self.params.add("model_dir", self.model_dir)
        wandb.config.update(self.params.dict)
        training_data = self.base.setup_data(mode='train')
        validation_data = self.base.setup_data(mode='valid')
        model, optimizer = self.base.setup_train()
        params = self.params
        valid_losses = []
        valid_rouges = []
        valid_bleus = []
        valid_fun_corrs = []
        for epoch_i in range(params.epoch):
            print('[ Epoch', epoch_i+1, ']')
            # train
            train_loss,  train_rouge, train_bleu, train_func_corr = self.base.train_epoch(model, training_data, training_data.dataset.max_src_oovs, optimizer)
            # eval
            valid_loss, valid_rouge, valid_bleu, valid_func_corr = self.base.eval_epoch(model, validation_data, validation_data.dataset.max_src_oovs)
            #print("***************",valid_loss)
            # log
            wandb.log({ "Training loss: ": train_loss,
                        "Training rouge: ": train_rouge, 
                        "Training bleu: ": train_bleu, 
                        "Training func_corr":train_func_corr,
                        "Validation loss: ": valid_loss,
                        "Validation rouge: ": valid_rouge,
                        "Validation bleu: ": valid_bleu,
                        "Validation func_corr: ": valid_func_corr})

            checkpoint = {'epoch': epoch_i, 'settings': params, 'model': model.state_dict()}

            if params.save_mode:
                valid_losses += [valid_loss]
                valid_rouges += [valid_rouge]
                valid_bleus += [valid_bleu]
                valid_fun_corrs += [valid_func_corr]
                if params.save_mode == 'best':
                    model_name = params.encoder_language + "_" + params.save_mode + '_loss_model.chkpt'
                    if valid_loss <= min(valid_losses):
                        torch.save(checkpoint, os.path.join(self.model_dir,model_name))
                        print('    - [Info] The checkpoint file has been updated by loss.')
                    model_name_rouge = params.encoder_language + "_"  + params.save_mode + '_rouge_model.chkpt'
                    if valid_rouge >= max(valid_rouges):
                        torch.save(checkpoint, os.path.join(self.model_dir,model_name_rouge))
                        print('    - [Info] The checkpoint file has been updated by rouge.')
                    model_name_bleu = params.encoder_language + "_"  + params.save_mode + '_bleu_model.chkpt'
                    if valid_bleu >= max(valid_bleus):
                        torch.save(checkpoint, os.path.join(self.model_dir,model_name_bleu))
                        print('    - [Info] The checkpoint file has been updated by bleu.')
                    model_name_func_corr = params.encoder_language + "_"  + params.save_mode + '_func_corr_model.chkpt'
                    if valid_func_corr >= max(valid_fun_corrs):
                        torch.save(checkpoint, os.path.join(self.model_dir,model_name_func_corr))
                        print('    - [Info] The checkpoint file has been updated by bleu.')

    def basic_decode(self, best_model_on_validation, data_file_prefix): 
        self.base = Procedure(self.params)
        test_data = self.base.setup_data(mode=data_file_prefix)
        decode_processor = Search(self.params, best_model_on_validation, test_data, data_file_prefix)
        result_dict = decode_processor.decode()
        print(result_dict)

    def decode(self, model_path, data_file_prefix="test"):
        if ";" in model_path:
            for path in model_path.split(";"):
                param_path = os.path.join(path, "params.json")
                self.params = Params(param_path)
                model_names = [self.params.encoder_language+"_best_func_corr_model.chkpt", self.params.encoder_language+"_best_bleu_model.chkpt", self.params.encoder_language+"_best_rouge_model.chkpt"]
                for m_name in model_names:
                    best_model_on_validation = os.path.join(path, m_name)
                    self.basic_decode(best_model_on_validation, data_file_prefix)
        else:
            param_path = os.path.join(model_path, "params.json")
            self.params = Params(param_path)
            model_names = [self.params.encoder_language+"_best_func_corr_model.chkpt", self.params.encoder_language+"_best_bleu_model.chkpt", self.params.encoder_language+"_best_rouge_model.chkpt"]
            for m_name in model_names:
                best_model_on_validation = os.path.join(model_path, m_name)
                self.basic_decode(best_model_on_validation, data_file_prefix)

if __name__ == '__main__':
    init_seeds()
    fire.Fire(Main)
#python -m run train --param-path params.json