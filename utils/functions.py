import pdb
import time
import random
import numpy as np
import logging

import torch


def common_process_opt(opt):
    if opt.seed > 0:
        set_seed(opt.seed)

    return opt


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def time_since(start_time):
    return time.time() - start_time


def read_tokenized_src_file(path, remove_title_eos=True):
    """
    读测试文件
    read tokenized source text file and convert them to list of list of words
    :param path:
    :param remove_title_eos: concatenate the words in title and content
    :return: data, a 2d list, each item in the list is a list of words of a src text, len(data) = num_lines
    """
    tokenized_train_src = []
    tokenized_train_tlt=[]
    for line_idx, src_line in enumerate(open(path, 'r')):
        # process source line
        title_and_context = src_line.strip().split('<eos>')
        if len(title_and_context) == 1:  # it only has context without title
            [context] = title_and_context
            src_word_list = context.strip().split(' ')
            title_word_list=""
        elif len(title_and_context) == 2:
            [title, context] = title_and_context
            title_word_list = title.strip().split(' ')
            context_word_list = context.strip().split(' ')
            if remove_title_eos:
                # src_word_list = title_word_list + context_word_list
                src_word_list =  context_word_list

            else:
                # src_word_list = title_word_list + ['<eos>'] + context_word_list
                src_word_list =context_word_list
        else:
            raise ValueError("The source text contains more than one title")
        # Append the lines to the data
        tokenized_train_src.append(src_word_list)
        tokenized_train_tlt.append(title_word_list)


    return tokenized_train_src,tokenized_train_tlt


def read_tokenized_trg_file(path):
    """
    read tokenized target text file and convert them to list of list of words
    :param path:
    :return: data, a 3d list, each item in the list is a list of target, each target is a list of words.
    """
    data = []
    with open(path) as f:
        for line in f:
            trg_list = line.strip().split(';')  # a list of target sequences
            trg_word_list = [trg.split(' ') for trg in trg_list]
            data.append(trg_word_list)
    return data


def read_src_and_trg_files(src_file, trg_file, is_train, remove_title_eos=True):
    tokenized_train_src = []
    tokenized_train_trg = []
    tokenized_train_tlt = []

    filtered_cnt = 0
    # zip 将对象打包成元组，enumerate会返回索引和参数值， 
    for line_idx, (src_line, trg_line) in enumerate(zip(open(src_file, 'r'), open(trg_file, 'r'))):
        # process source line
        if (len(src_line.strip()) == 0) and is_train:
            continue                     # 如果为空行，不处理
        title_and_context = src_line.strip().split('<eos>')
        
        if len(title_and_context) == 1:        # it only has context without title,没有标题
            [context] = title_and_context
            src_word_list = context.strip().split(' ')
        elif len(title_and_context) == 2:           # 有标题
            [title, context] = title_and_context
            
            title_word_list = title.strip().split(' ')
            context_word_list = context.strip().split(' ')
            #shen
            # title = title_and_context[0]                # 获取标题
            
            if remove_title_eos:        # True
                # src_word_list = title_word_list + context_word_list
                src_word_list =context_word_list
                tlt_word_list=title_word_list


            else:
                src_word_list = title_word_list + ['<eos>'] + context_word_list
        else:
            raise ValueError("源文本包含多个标题")     #The source text contains more than one title
        # process target line
        trg_list = trg_line.strip().split(';')  # a list of target sequences
        trg_word_list = [trg.split(' ') for trg in trg_list]
        # If it is training data, ignore the line with source length > 400 or target length > 60
        # 训练数据，忽略原始长度大于400， 目标长度大于60 的数据
        #删除没有标题的数据
        if is_train:
            if len(src_word_list) > 400 or len(trg_word_list) > 14:
                filtered_cnt += 1
                continue
            elif len(tlt_word_list)==0:
                filtered_cnt += 1
                continue
           

        # Append the lines to the data
        tokenized_train_src.append(src_word_list)
        tokenized_train_trg.append(trg_word_list)
        tokenized_train_tlt.append(tlt_word_list)

    assert len(tokenized_train_src) == len(
        tokenized_train_trg)==len(tokenized_train_tlt), 'the number of records in source and target are not the same'

#
    # logging.info("%d rows filtered" % filtered_cnt)
    logging.info("过滤的行数： %d" % filtered_cnt)

    # tokenized_train_pairs = list(zip(tokenized_train_src, tokenized_train_trg))
    tokenized_train_pairs = list(zip(tokenized_train_src, tokenized_train_trg,tokenized_train_tlt))
    return tokenized_train_pairs

