import logging
import pdb
from tkinter import Variable

import torch
from torch.utils.data import DataLoader

from pykp.utils.io import KeyphraseDataset
import pdb

def load_vocab(opt):
    # load vocab
    logging.info("Loading vocab from disk: %s" % opt.vocab)
    vocab = torch.load(opt.vocab + '/vocab.pt', 'wb')
    # assign vocab to opt
    opt.vocab = vocab
    logging.info('#(vocab)=%d' % len(vocab["word2idx"]))
    logging.info('#(vocab used)=%d' % opt.vocab_size)

    return vocab


def build_data_loader(data, opt, shuffle=True, load_train=True):

    #data=train_data  即.pt数据

    # print(type(data))     #list
    # exit()
    keyphrase_dataset = KeyphraseDataset.build(examples=data, opt=opt, load_train=load_train)
    

    if not opt.one2many:
        collect_fn = keyphrase_dataset.collate_fn_one2one     # 返回tuple
    elif opt.fix_kp_num_len:
        collect_fn = keyphrase_dataset.collate_fn_fixed_tgt      # one2set
    else:
        collect_fn = keyphrase_dataset.collate_fn_one2seq
    

   # collate_fn  以循环形式处理函数， 输出真实的样本数据
    #返回：src, src_lens, src_mask, src_oov, oov_lists, src_str, \
      #         trg_str, trg, trg_oov, trg_lens, trg_mask, original_indices, tlt_str,tlt,tlt_lens,tlt_mask

    data_loader = DataLoader(dataset=keyphrase_dataset, collate_fn=collect_fn,
                             num_workers=opt.batch_workers,
                             batch_size=opt.batch_size, shuffle=shuffle)
    #batch_size  自定义
    return data_loader


def load_data_and_vocab(opt, load_train=True):
    vocab = load_vocab(opt)
    # constructor data loader
    logging.info("Loading train and validate data from '%s'" % opt.data)
    if opt.one2many:
        data_path = opt.data + '/%s.one2many.pt'
    else:
        data_path = opt.data + '/%s.one2one.pt'        #数据集字典位置

    if load_train:
        # load training dataset
        #list
        train_data = torch.load(data_path % "train", 'wb')       #opt.data + '/train.one2many.pt'
        
        # print(type(train_data[0]))
        # # print(train_data[0])
        # for key in train_data[0].keys():
        #     print(key)
            # print(train_data[0][key])
        # exit()
        #shuffle=True

        train_loader = build_data_loader(data=train_data, opt=opt, shuffle=False, load_train=True)
       
            #here goes neural nets part
        # logging.info('#(train data size: #(batch)=%d' % (len(train_loader)))
        logging.info('#(train data length: #(batch)=%d' % (len(train_loader)))

        # load validation dataset
        valid_data = torch.load(data_path % "valid", 'wb')
        valid_loader = build_data_loader(data=valid_data,  opt=opt, shuffle=False, load_train=True)
        logging.info('#(valid data size: #(batch)=%d' % (len(valid_loader)))
        return train_loader, valid_loader, vocab

    else:
        test_data = torch.load(data_path % "test", 'wb')
        test_loader = build_data_loader(data=test_data, opt=opt, shuffle=False, load_train=False)
        logging.info('#(test data size: #(batch)=%d' % (len(test_loader)))
        return test_loader, vocab
