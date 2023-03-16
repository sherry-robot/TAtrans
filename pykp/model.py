import torch.nn as nn

import pykp.utils.io as io
from pykp.decoder.transformer import TransformerSeq2SeqDecoder
from pykp.encoder.transformer import TransformerSeq2SeqEncoder
from pykp.modules.position_embed import get_sinusoid_encoding_table
# from pykp.encoder.transformer import TitleAtteion
from pykp.encoder.transformer import title_attention_merge



class Seq2SeqModel(nn.Module):
    """Container module with an encoder, decoder, embeddings."""

    def __init__(self, opt):
        """Initialize model."""
        super(Seq2SeqModel, self).__init__()
        #（size,512,   ）,  pad, 设置需要pad的符号

        embed = nn.Embedding(opt.vocab_size, opt.word_vec_size, opt.vocab["word2idx"][io.PAD_WORD])
        self.init_emb(embed)
        #获得位置嵌入
        pos_embed = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(3000, opt.word_vec_size, padding_idx=opt.vocab["word2idx"][io.PAD_WORD]), freeze=True)
        self.encoder = TransformerSeq2SeqEncoder.from_opt(opt, embed, pos_embed)
        self.title_attention_merge=title_attention_merge()
        # self.enc_attn=encode_attn(opt.d_model)
        self.decoder = TransformerSeq2SeqDecoder.from_opt(opt, embed, pos_embed)

    def init_emb(self, embed):
        """Initialize weights."""
        initrange = 0.1
        #初始化参数， 均匀分布
        embed.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_lens, input_tgt, src_oov, max_num_oov, src_mask,tlt, tlt_lens, tlt_mask):
        """
        :param src: a LongTensor containing the word indices of source sentences, [batch, src_seq_len], with oov words replaced by unk idx
        :param src_lens: a list containing the length of src sequences for each batch, with len=batch, with oov words replaced by unk idx
        :param trg: a LongTensor containing the word indices of target sentences, [batch, trg_seq_len]
        :param src_oov: a LongTensor containing the word indices of source sentences, [batch, src_seq_len], contains the index of oov words (used by copy)
        :param max_num_oov: int, max number of oov for each batch
        :param src_mask: a FloatTensor, [batch, src_seq_len]
        :return:
        """
        # Encoding
        title_embed, src_embed =self.encoder(src, src_lens, src_mask, tlt, tlt_lens, tlt_mask)
        # title_embed = self.encoder(tlt, tlt_lens, tlt_mask)
        encoder_outpput=self.title_attention_merge(src_embed, title_embed,src_mask)
        state = self.decoder.init_state(encoder_outpput, src_mask)
        decoder_dist_all, attention_dist_all = self.decoder(input_tgt, state, src_oov, max_num_oov)
        return decoder_dist_all, attention_dist_all


