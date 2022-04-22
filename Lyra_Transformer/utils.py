# -*- coding: utf-8 -*-
import random
import numpy as np
import torch
from torch import nn
import time
import os
import re
import parso
from parso.python import tokenize
version_info = parso.utils.parse_version_string("3.8")
from rouge import Rouge
from nltk.translate import bleu_score
from nltk.translate.bleu_score import corpus_bleu
from pylint import epylint as lint
os.environ['CUDA_VISIBLE_DEVICES']='1,2,3,4,5,6,7'
# init
def init_seeds():
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
        torch.cuda.manual_seed_all(123)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(123)

def init_lstm_weight(lstm):
    for param in lstm.parameters():
        if len(param.shape) >= 2: # weights
            init_ortho_weight(param.data)
        else: # bias
            init_bias(param.data)

def init_gru_weight(gru):
    for param in gru.parameters():
        if len(param.shape) >= 2: # weights
            init_ortho_weight(param.data)
        else: # bias
            init_bias(param.data)

def init_linear_weight(linear):
    init_xavier_weight(linear.weight)
    if linear.bias is not None:
        init_bias(linear.bias)

def init_normal_weight(w):
    nn.init.normal_(w, mean=0, std=0.01)

def init_uniform_weight(w):
    nn.init.uniform_(w, -0.1, 0.1)

def init_ortho_weight(w):
    nn.init.orthogonal_(w)

def init_xavier_weight(w):
    nn.init.xavier_normal_(w)

def init_bias(b):
    nn.init.constant_(b, 0.)


# evaluation metrics
def get_bleu4_score(hyps_list, gold_list, tokenizer):
    b_score = corpus_bleu(
        [[tokenizer.tokenize(ref)] for ref in gold_list],
        [tokenizer.tokenize(pred) for pred in hyps_list],
        smoothing_function = bleu_score.SmoothingFunction(epsilon=1e-12).method1, 
        weights=(0.25, 0.25, 0.25, 0.25))
    return b_score

def get_rouge_dict(hyps_list, gold_list):
    rouge = Rouge()
    result_dict = rouge.get_scores(hyps_list, gold_list, avg=True)
    return result_dict

def get_var_replacing(code_string, repalce_string):
    version_info = parso.utils.parse_version_string("3.8")
    var_dict = {}
    token_list = []
    var_index = 0
    for i in tokenize.tokenize(code_string, version_info):
        if not repalce_string:
            # print(i)
            if i.type == tokenize.NAME:
                if i.string in var_dict.keys():
                    token_list.append(var_dict[i.string])
                else:
                    var = "var_"+str(var_index)
                    var_dict[i.string] = var
                    token_list.append(var)
                    var_index+=1
            elif i.type == tokenize.STRING and re.findall(r"( FROM )|( from )", i.string)!=[]:
                try:
                    sql_parsed = sqlparse(i.string[1:-1])
                except:
                    sql_parsed = i.string
                token_list.append(sql_parsed)
            else:
                token_list.append(i.string)
        else:
            if i.type == tokenize.NAME or (i.type == tokenize.STRING and re.findall(r"( FROM )|( from )", i.string)!=[]):
                if i.string in var_dict.keys():
                    token_list.append(var_dict[i.string])
                else:
                    var = "var_"+str(var_index)
                    var_dict[i.string] = var
                    token_list.append(var)
                    var_index+=1
            else:
                token_list.append(i.string)        
    return token_list


def get_func_correctness(hyps_list, gold_list, repalce_string=False, need_index=False):
    ast_match_num = 0
    index = 0
    index_list = []
    for i, j  in zip(hyps_list, gold_list):
        if '<unk>' not in i:
            i, j = get_var_replacing(i, repalce_string), get_var_replacing(j, repalce_string)
            if i == j:
                ast_match_num+=1
                index_list.append(index)
        index+=1
    print("Number of AST matching", ast_match_num)
    print("Accuration of AST matching", ast_match_num/len(hyps_list))
    if need_index==True:
        return ast_match_num/len(hyps_list), " ".join([str(k) for k in index_list])
    else:
        return ast_match_num/len(hyps_list)

def embedded_ast_matching(hyps_list, gold_list, need_index =False):
    version_info = parso.utils.parse_version_string("3.8")
    def get_sql_parser(s):   
        sqls = [] 
        for t in tokenize.tokenize(s, version_info):
            if t.type == tokenize.STRING and re.findall(r"( FROM )|( from )", t.string)!=[]:
                try:
                    sql_parsed = sqlparse(t.string[1:-1])
                except:
                    sql_parsed = t.string
                sqls.append(sql_parsed)
        return sqls
    m_num = 0
    index_list = []
    index = 0
    for i, j in zip(hyps_list, gold_list):
        if get_sql_parser(i)!=[] and get_sql_parser(i) == get_sql_parser(j):
            # print(i, j)
            # print("---------------")
            m_num+=1
            index_list.append(index)
        index+=1
    if need_index==True:
        return m_num/len(hyps_list), " ".join([str(k) for k in index_list])
    else:
        return m_num/len(hyps_list)
    
def code_staticAnaylsis(code, id):
    cur_time = str(time.time()).replace(".","")
    with open(cur_time+str(id)+".py",'w') as f:
        f.write("# pylint: disable=E1101\n")
        f.write(code)
    (pylint_stdout, pylint_stderr) = lint.py_run(cur_time+str(id)+".py -s yes", return_std=True)
    os.remove(cur_time+str(id)+".py")
    pylint_stdout_str = pylint_stdout.read()
    # pylint_stderr_str = pylint_stderr.read()
    if "E0" in pylint_stdout_str or "E1" in pylint_stdout_str:
        return True
    return False

def get_executable_rate(hyps_list):
    executable_wrong_num = 0
    for i in range(len(hyps_list)):
        if '<unk>' not in hyps_list[i]:
            if code_staticAnaylsis(hyps_list[i].replace("\t","    "), i):
                executable_wrong_num+=1  
        else:
            executable_wrong_num+=1
    return (len(hyps_list) - executable_wrong_num)/len(hyps_list)

def sqlparse(s):
    s = s.replace("\n"," ").replace("\\"," ")
    # split index
    se = re.findall(r"(SELECT )|(select )", s)
    fr = re.findall(r"( FROM )|( from )", s)
    wh = re.findall(r"( WHERE )|( where )", s)
    try :
        assert len(se)==len(fr)==1
    except:
        return s
    s_index = s.index(se[0][0] if se[0][0]!="" else se[0][1])
    f_index = s.index(fr[0][0] if fr[0][0]!="" else fr[0][1])
    if wh!=[]:
        w_index = s.index(wh[0][0] if wh[0][0]!="" else wh[0][1])
    else:
        w_index = 9999

    # split tokens
    query_tokens = []
    from_toknes = []
    where_tokens = []
    for i in tokenize.tokenize(s, version_info):
        # print(i)
        if i.type in [tokenize.INDENT, tokenize.DEDENT, tokenize.ENDMARKER]:
            continue
        if i.start_pos[1] < f_index:
            query_tokens.append(i)
        elif i.start_pos[1] < w_index:
            from_toknes.append(i)
        else:
            where_tokens.append(i)

    # condition list
    if len(where_tokens)!=0:
        #where_list = [where_tokens[0].string]
        where_list = []
        temp = []
        temp_op = ''
        for i in where_tokens[1:]:
            if i.type == tokenize.OP and i.string == ',':
                temp.append({"OP":i.string})
            elif i.type == tokenize.OP and i.string != ':':
                temp.append({"OP":i.string})
                temp_op = i.string
            elif i.type != tokenize.OP:
                if temp_op!='':
                    temp.append({"PARAMETER":i.string})
                    temp_op=''
                else:
                    temp.append({"COLUMN":i.string})
        cond ={}
        cond_list = []
        # print("temp: ",temp)
        for i in range(len(temp)):
            # print(temp[i])
            if (i+1)%4==0:
                cond_list.append(cond)
                # cond_list.append(temp[i])
                cond = {}
            else:
                cond = dict(cond, **temp[i])
        if len(cond)!=0:
            cond_list.append(cond)
        # where_list.append(cond_list)
        where_list = {"CONDITION":cond_list}
    else:
        where_list=[]

    # from list
    # from_lsit = [from_toknes[0].string, [from_toknes[1].string, where_list]]
    from_lsit = {"TABLE":from_toknes[1].string}

    # query list
    # query_list = [query_tokens[0].string]
    query_list = []
    temp = []
    op_list = []
    col_list = []
    for i in query_tokens[1:]:
        if i.type!=tokenize.OP:
            temp.append(i.string)
        else:
            op_list.append(i.string)
    if ',' in op_list:
        for i in temp:
            col_list.append({"COLUMN":i})
    elif "(" in op_list and ")" in op_list and len(temp)<=2:
        col_list.append({"AGG":temp[0]})
        if len(temp)==1:
            col_list.append({"COLUMN":op_list[1]})
        else:
            col_list.append({"COLUMN":temp[1]})
    elif len(op_list)==0 and len(temp)==1:
        col_list.append({"COLUMN":temp[0]})
    elif op_list[0]=="*" and len(temp)==0:
        col_list.append({"COLUMN":op_list[0]})
    else:
        # print(col_list, temp)
        return "Error"
    
    query_list = [{"COLUMNS":col_list}, from_lsit, where_list]

    return query_list