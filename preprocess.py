import argparse
import logging
import os
from collections import Counter
import torch

import config
import pykp.utils.io as io
from utils.functions import read_src_and_trg_files
import pdb
from transformers import BertTokenizer, BertModel




def bert_vocab(opt):
    word2idx = dict()
    idx2word = dict()
    # trg_frequent = Counter()
    bert_vocab=opt.vocab+"/vocab-bert.txt"

    file = open(bert_vocab, 'r', encoding='utf-8')
    for id, token in enumerate(file):
        token = token.replace("\n", "")
        tokenlist = []
        tokenlist.append(token)
        word2idx[token] = id
        idx2word[id] = token
        # trg_frequent.update(tokenlist)
#trg_frequent
    return word2idx,idx2word



def tokenids2word(tokenids,tokenizer):

    retokens=tokenizer.convert_ids_to_tokens(tokenids)
    word=""

    for token in retokens:
        if(token.startswith("##") and token!=retokens[0] ):
           
            word+=token[2:]

        else:
            word+=token
    return word




def build_bertvocab(tokenized_src_trg_pairs,opt):
    # word2idx, idx2word = bert_vocab(opt)

    vocab_path=opt.vocab+"/vocab-bert.txt"
    tokendict={}
    fopen=open(vocab_path, 'r',encoding="utf-8")
    filecontent = open(vocab_path, 'r',encoding="utf-8").read()
    bert_path: str=r"bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(bert_path)

    token_freq_counter = Counter()
    for src_word_list, trg_word_lists,tlt_word_list in tokenized_src_trg_pairs:
        token_freq_counter.update(src_word_list)    # 更新counter字典
        token_freq_counter.update(tlt_word_list)
        for word_list in trg_word_lists:
            token_freq_counter.update(word_list)
    #添加特殊token到词表

    special_tokens = [io.PAD_WORD, io.UNK_WORD, io.BOS_WORD, io.EOS_WORD, io.SEP_WORD, io.PEOS_WORD,
                      io.NULL_WORD]
    
    specialtoken_dict={'additional_special_tokens':[io.PAD_WORD, io.UNK_WORD, io.BOS_WORD, io.EOS_WORD, io.SEP_WORD, io.PEOS_WORD,
                      io.NULL_WORD]}

    num_special_toks=tokenizer.add_special_tokens(specialtoken_dict)
    # for spt in special_tokens:
    #     tokens=tokenizer.tokenize(spt) 
    #     print(tokens)

    for s_t in special_tokens:
        if s_t in token_freq_counter:
            del token_freq_counter[s_t]
        
    word2idx = dict()
    idx2word = dict()
    # word2ids=dict()
    word2order=dict()
    # word2bertids=dict()
    # bertids2word=dict()
  

    sorted_word2idx = sorted(token_freq_counter.items(), key=lambda x: x[1], reverse=True)
    # print(type(sorted_word2idx))
    
    sorted_words = [x[0] for x in sorted_word2idx]

    for idx,word in enumerate(sorted_words):
        tokens=tokenizer.tokenize(word)
        bert_id = tokenizer.convert_tokens_to_ids(tokens)
        bert_tuple=tuple(bert_id)
        word2order[word]=idx
        # word2ids[word] = bert_id 
        word2idx[word]=bert_tuple
        idx2word[bert_tuple]=word

        # word2idx[word]=bert_id
        # reword= tokenids2word(bert_id,tokenizer)
        # print(word)
        # print(reword)

    # vocab = {"word2idx": word2idx, "idx2word": idx2word, "counter": token_freq_counter}
    vocab={"word2order": word2order,"word2idx":word2idx , "idx2word":idx2word , "counter": token_freq_counter}
    
    # print(vocab["word2order"])

    # for key in vocab["word2idx"]:
    #     id=vocab["word2idx"][key]
    #     word =vocab["idx2word"][id]
    #     pdb.set_trace()
    #     print(key,id,word)
      
    return vocab


# 创建词表
def build_vocab(tokenized_src_trg_pairs,opt):
   
    token_freq_counter = Counter()
    for src_word_list, trg_word_lists,tlt_word_list in tokenized_src_trg_pairs:
        token_freq_counter.update(src_word_list)    # 更新counter字典
        token_freq_counter.update(tlt_word_list)
        for word_list in trg_word_lists:
            token_freq_counter.update(word_list)

    # Discard special tokens if already present
    #忽略特殊token
    special_tokens = [io.PAD_WORD, io.UNK_WORD, io.BOS_WORD, io.EOS_WORD, io.SEP_WORD, io.PEOS_WORD,
                      io.NULL_WORD]
    num_special_tokens = len(special_tokens)

    for s_t in special_tokens:
        if s_t in token_freq_counter:
            del token_freq_counter[s_t]

    word2idx = dict()
    idx2word = dict()
    for idx, word in enumerate(special_tokens):
        word2idx[word] = idx
        idx2word[idx] = word
        #词频从小到大排序，字典, 
    sorted_word2idx = sorted(token_freq_counter.items(), key=lambda x: x[1], reverse=True)
    # print(type(sorted_word2idx))
    
    sorted_words = [x[0] for x in sorted_word2idx]

    for idx, word in enumerate(sorted_words):
        if(word!="<peos>"):
            word2idx[word] = idx + num_special_tokens
            idx2word[idx + num_special_tokens] = word

    # for idx, word in enumerate(sorted_words):
    #     idx2word[idx + num_special_tokens] = word

    vocab = {"word2idx": word2idx, "idx2word": idx2word, "counter": token_freq_counter}
    
    print(len(vocab["word2idx"]))
    print(len(vocab["counter"]))
   
    
    return vocab


def main(opt):
    # Tokenize train_src and train_trg, return a list of tuple, (src_word_list, [trg_1_word_list, trg_2_word_list, ...])
   
    tokenized_train_pairs = read_src_and_trg_files(opt.train_src, opt.train_trg, is_train=True,
                                                   remove_title_eos=opt.remove_title_eos)
    
    # tokenized_train_src, tokenized_train_trg,tokenized_train_tlt
    tokenized_valid_pairs = read_src_and_trg_files(opt.valid_src, opt.valid_trg, is_train=False,
                                                   remove_title_eos=opt.remove_title_eos)
    #创建词表入口
    # vocab=build_bertvocab(tokenized_train_pairs,opt)
    # exit()
    vocab = build_vocab(tokenized_train_pairs,opt)
    opt.vocab = vocab
    logging.info("Dumping dict to disk: %s" % opt.save_data_dir + '/vocab.pt')
    
    torch.save(vocab, open(opt.save_data_dir + '/vocab.pt', 'wb'))

    if not opt.one2many:
        # saving  one2one datasets
        train_one2one = io.build_dataset(tokenized_train_pairs, opt, mode='one2one')
        logging.info("Dumping train one2one to disk: %s" % (opt.save_data_dir + '/train.one2one.pt'))
        torch.save(train_one2one, open(opt.save_data_dir + '/train.one2one.pt', 'wb'))
        len_train_one2one = len(train_one2one)
        del train_one2one

        valid_one2one = io.build_dataset(tokenized_valid_pairs, opt, mode='one2one')

        logging.info("Dumping valid to disk: %s" % (opt.save_data_dir + '/valid.one2one.pt'))
        torch.save(valid_one2one, open(opt.save_data_dir + '/valid.one2one.pt', 'wb'))

        logging.info('#pairs of train_one2one  = %d' % len_train_one2one)
        logging.info('#pairs of valid_one2one  = %d' % len(valid_one2one))
    else:
        # saving  one2many datasets
        train_one2many = io.build_dataset(tokenized_train_pairs, opt, mode='one2many')
        logging.info("Dumping train one2many to disk: %s" % (opt.save_data_dir + '/train.one2many.pt'))
        torch.save(train_one2many, open(opt.save_data_dir + '/train.one2many.pt', 'wb'))
        len_train_one2many = len(train_one2many)
        del train_one2many

       #返回数据对
        valid_one2many = io.build_dataset(tokenized_valid_pairs, opt, mode='one2many')

        logging.info("Dumping valid to disk: %s" % (opt.save_data_dir + '/valid.one2many.pt'))
        torch.save(valid_one2many, open(opt.save_data_dir + '/valid.one2many.pt', 'wb'))

        logging.info('#pairs of train_one2many = %d' % len_train_one2many)
        logging.info('#pairs of valid_one2many = %d' % len(valid_one2many))
    logging.info('Done!')


#作用： 创建词表，
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='preprocess.py',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    #创建parser实例
    config.vocab_opts(parser)        # 添加参数
    config.preprocess_opts(parser)
    opt = parser.parse_args()          #解析参数

    logging = config.init_logging(log_file=opt.log_path + "/output.log", stdout=True)

    if not opt.one2many:
        test_exists = os.path.join(opt.save_data_dir, "train.one2one.pt")
    else:
        test_exists = os.path.join(opt.save_data_dir, "train.one2many.pt")
    if os.path.exists(test_exists):
        logging.info("file exists %s, exit! " % test_exists)
   

    opt.train_src = opt.data_dir + '/train_src.txt'
    opt.train_trg = opt.data_dir + '/train_trg.txt'
    opt.valid_src = opt.data_dir + '/valid_src.txt'
    opt.valid_trg = opt.data_dir + '/valid_trg.txt'
    main(opt)
