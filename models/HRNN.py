import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *
import math
import numpy as np
import json
from collections import OrderedDict
from .CaptionModel import CaptionModel

class Word_LSTM(nn.Module):

    def __init__(self, opt, use_maxout=False):
        super(Word_LSTM, self).__init__()
        self.lang_drop = opt.lang_drop
        self.epsilon = opt.epsilon
        self.word_rnn_size = opt.word_rnn_size
        self.att_lstm = nn.LSTMCell(opt.word_encoding_size + opt.word_rnn_size * 2, opt.word_rnn_size)
        self.lang_lstm = nn.LSTMCell(opt.word_rnn_size * 3, opt.word_rnn_size)

        self.confidence = nn.Sequential(nn.Linear(opt.word_rnn_size, opt.word_rnn_size),
                                        nn.ReLU(),
                                        nn.Linear(opt.word_rnn_size, 1),
                                        nn.Sigmoid())
        self.tanh = nn.Tanh()
        self.update = nn.Sequential(nn.Linear(opt.word_rnn_size, opt.word_rnn_size),
                                              nn.ReLU())
    def forward(self, xt,it, fc_feats, visual_value, visual_key,lang_value, lang_key, kv_attention, update_kv,max_att_step,state,kv_state):
        batch_size = fc_feats.size()[0]
        word_exist = (it>0).data.float()

        accum_conf = Variable(fc_feats.data.new(batch_size, 1).zero_())
        self.att_step = Variable(fc_feats.data.new(batch_size).zero_())
        self.att_cost = Variable(fc_feats.data.new(batch_size).zero_())
        selector = Variable(fc_feats.data.new(batch_size,1).fill_(1)).byte()
        h_att_ = Variable(fc_feats.data.new(batch_size, self.word_rnn_size).zero_())
        h_lang_ = Variable(fc_feats.data.new(batch_size, self.word_rnn_size).zero_())
        c_lang_ = Variable(fc_feats.data.new(batch_size, self.word_rnn_size).zero_())
        output_ = Variable(fc_feats.data.new(batch_size, self.word_rnn_size).zero_())

        c_att_ = Variable(fc_feats.data.new(batch_size, self.word_rnn_size).zero_())


        prev_h = state[0][-1]
        for i in range(max_att_step):

            att_lstm_input = torch.cat([prev_h, fc_feats, xt], 1)
            h_att, c_att = self.att_lstm(att_lstm_input, (state[0][0], state[1][0]))

            visual_context, lang_context, kv_state = kv_attention(h_att, xt, visual_key, visual_value,lang_key, lang_value, kv_state)
            lang_lstm_input = torch.cat([visual_context,lang_context, h_att], 1)  # [batch,512×2]
            h_lang, c_lang = self.lang_lstm(lang_lstm_input, (state[0][1], state[1][1]))
            output = F.dropout(h_lang, self.lang_drop, self.training)
            visual_key, lang_key = update_kv(output, visual_key, visual_value, lang_key, lang_value)

            p_ = self.confidence(h_lang)

            beta = p_ * (1 - accum_conf)
            accum_conf += beta * selector.float()
            h_att_ += beta * h_att * selector.float()
            c_att_ += beta * c_att * selector.float()
            h_lang_ += beta * h_lang * selector.float()
            c_lang_ += beta * c_lang * selector.float()
            output_ += beta * output * selector.float()
            self.att_cost += ((i + 1) * (1 - p_)) * selector.float().squeeze(1)
            selector = (accum_conf < 1 - self.epsilon).data * selector

            if not selector.any():
                break

        h_att_ /= accum_conf
        c_att_ /= accum_conf
        h_lang_ /= accum_conf
        c_lang_ /= accum_conf
        output_ /= accum_conf
        state = (
        torch.stack([h_att_, h_lang_]), torch.stack([c_att_, c_lang_]))  # ([2,batch,512],[2,batch,512])
        return output_,visual_key, lang_key,self.att_cost*word_exist, state,kv_state


class Language_Attention(nn.Module):
    def __init__(self, opt):
        super(Language_Attention, self).__init__()
        self.dense_hidden_size = opt.caption_rnn_size
        self.rnn_size = opt.word_rnn_size
        self.att_hid_size = opt.att_hid_size

        self.h2att = nn.Linear(self.rnn_size, self.att_hid_size)
        self.alpha_net = nn.Linear(self.att_hid_size, 1)

    def forward(self, h, l_value, l_key,v_weight):
        att_size = l_value.numel() // l_value.size(0) // l_value.size(-1)  # 50
        att_h = self.h2att(h)
        att_h = att_h.unsqueeze(1).expand_as(l_key)
        dot = l_key + att_h
        dot = F.tanh(dot)
        dot = dot.view(-1, self.att_hid_size)
        dot = self.alpha_net(dot)
        dot = dot.view(-1, att_size)

        weight = F.softmax(dot, dim=1) * v_weight
        att_res = torch.bmm(weight.unsqueeze(1), l_value).squeeze(1)
        return att_res,weight

class Visual_Attention(nn.Module):
    def __init__(self, opt):
        super(Visual_Attention, self).__init__()
        self.rnn_size = opt.word_rnn_size
        self.att_hid_size = opt.att_hid_size

        self.h2att = nn.Linear(self.rnn_size, self.att_hid_size)
        self.alpha_net = nn.Linear(self.att_hid_size, 1)

    def forward(self, h, att_feats, p_att_feats, att_masks=None):
        att_size = att_feats.numel() // att_feats.size(0) // att_feats.size(-1)
        att = p_att_feats.view(-1, att_size, self.att_hid_size)

        att_h = self.h2att(h)
        att_h = att_h.unsqueeze(1).expand_as(att)
        dot = att + att_h
        dot = F.tanh(dot)
        dot = dot.view(-1, self.att_hid_size)
        dot = self.alpha_net(dot)
        dot = dot.view(-1, att_size)

        weight = F.softmax(dot, dim=1)
        if att_masks is not None:
            weight = weight * att_masks.view(-1, att_size).float()
            weight = weight / weight.sum(1, keepdim=True)  # normalize to 1

        att_feats_ = att_feats.view(-1, att_size, att_feats.size(-1))  # batch * att_size * att_feat_size
        att_res = torch.bmm(weight.unsqueeze(1), att_feats_).squeeze(1)  # batch * att_feat_size

        return att_res,weight

class HRNN(CaptionModel):
    def __init__(self, opt):
        super(HRNN,self).__init__()
        self.opt = opt
        self.info = json.load(open(self.opt.input_json))
        self.paragraph_i2w = self.info['ix_to_word']
        self.vocab_size = len(self.paragraph_i2w)
        self.max_att_step = opt.max_att_step
        self.feat_drop = opt.feat_drop
        self.lang_feat = opt.lang_drop
        self.word_rnn_size = opt.word_rnn_size
        self.sent_max = opt.sen_max
        self.word_max = opt.word_max
        self.ss_prob = 0.0
        self.word_encoding_size = opt.word_encoding_size
        #  Sentence RNN:
        self.feats_dim = opt.feats_dim
        self.sen_rnn_size = opt.sen_rnn_size
        self.feats_pool_dim = opt.feats_pool_dim
        self.att_hid_size = opt.att_hid_size
        self.sen_lstm = nn.LSTMCell(self.feats_pool_dim, self.sen_rnn_size)
        self.feats_pool = nn.Linear(self.feats_dim,self.feats_pool_dim)
        self.feat_topic_embed = nn.Sequential(self.feats_pool, nn.ReLU(),)
        self.pre_prob = nn.Linear(self.sen_rnn_size, 2)

        # Word RNN
        self.topic_size = opt.topic_dim
        self.topic_layer = nn.Sequential(nn.Linear(self.sen_rnn_size, self.sen_rnn_size),nn.ReLU(),
                                         nn.Linear(self.sen_rnn_size, self.word_rnn_size),nn.ReLU(),
                                         nn.Dropout(self.feat_drop)
                                        )
        self.Embedding = nn.Embedding(self.vocab_size, self.word_encoding_size)
        self.embed = nn.Sequential(self.Embedding,nn.ReLU(),
                                   nn.Dropout(self.lang_feat))
        self.word_lstm = Word_LSTM(opt)
        self.logit = nn.Linear(self.word_rnn_size, self.vocab_size-2)

        self.softmax = nn.Softmax(dim=1)

        # visual key-value setting
        self.kv_lstm = nn.LSTMCell(self.word_encoding_size+self.word_rnn_size, self.word_rnn_size)
        self.visual_att = Visual_Attention(opt)
        self.att_embed = nn.Sequential(
                nn.Linear(self.feats_dim, self.word_rnn_size),
                 nn.ReLU(),nn.Dropout(self.feat_drop))
        self.ctx2att = nn.Linear(self.word_rnn_size, self.att_hid_size)
        self.visual_add_gate = nn.Linear(self.word_rnn_size, self.word_rnn_size)
        self.visual_forget_gate = nn.Linear(self.word_rnn_size, self.word_rnn_size)
        self.sigmoid = nn.Sigmoid()

        # language key-value setting
        self.lang_att = Language_Attention(opt)
        self.lang_add_gate = nn.Linear(self.word_rnn_size, self.att_hid_size)
        self.lang_forget_gate = nn.Linear(self.word_rnn_size, self.att_hid_size)
        self.lang_embed = nn.Linear(self.word_rnn_size, self.word_rnn_size)

        self.confidence = nn.Sequential(nn.Linear(self.word_rnn_size, self.word_rnn_size),
                                        nn.ReLU(),
                                        nn.Linear(self.word_rnn_size, 1),
                                        nn.Sigmoid())


    def init_sent_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (Variable(weight.new(bsz, self.sen_rnn_size).zero_()),
                Variable(weight.new(bsz, self.sen_rnn_size).zero_()))
    def init_word_lstm2_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (Variable(weight.new(bsz, self.sen_rnn_size).zero_()),
                Variable(weight.new(bsz, self.sen_rnn_size).zero_()))
    def kv_attention(self,pre_hidden,pre_word,v_key,v_value,l_key,l_value,state):
        kv_input = torch.cat((pre_hidden, pre_word), dim=1)
        query, c = self.kv_lstm(kv_input, state)
        state = torch.stack([query, c])
        # process visual
        v_att_res, v_weight = self.visual_att(query, v_value, v_key)
        # process language
        l_att_res, l_weight = self.lang_att(query, l_value, l_key, v_weight)
        return v_att_res,l_att_res,state
    def updata_kv(self,h,v_key,v_value,l_key,l_value):
        # updata visual key
        _,v_weight = self.visual_att(h,v_value,v_key)
        v_add_weight = self.sigmoid(self.visual_add_gate(h))
        v_forget_weight = self.sigmoid(self.visual_forget_gate(h))
        v_add = torch.bmm(v_weight.unsqueeze(2), v_add_weight.unsqueeze(1))
        v_forget = torch.bmm(v_weight.unsqueeze(2), v_forget_weight.unsqueeze(1))
        v_key = v_key*(1-v_forget)+v_add

        _, l_weight = self.lang_att(h, l_value, l_key, v_weight)
        l_add_weight = self.sigmoid(self.lang_add_gate(h))
        l_forget_weight = self.sigmoid(self.lang_forget_gate(h))
        l_add = torch.bmm(l_weight.unsqueeze(2),
                          l_add_weight.unsqueeze(1))
        l_forget = torch.bmm(l_weight.unsqueeze(2),
                             l_forget_weight.unsqueeze(1))
        l_key = l_key * (1 - l_forget) + l_add
        return v_key, l_key

    def get_logprobs_state(self, it,topic_vector, visual_value, visual_key,lang_value, lang_key, kv_state, state):

        xt = self.embed(it+2)
        output, visual_key, lang_key, att_cost, state, kv_state = self.word_lstm(xt, it+2, topic_vector,
                                                                                 visual_value, visual_key, lang_value,
                                                                                 lang_key, self.kv_attention,
                                                                                 self.updata_kv,
                                                                                 self.max_att_step, state, kv_state)

        output = self.logit(output)
        logprobs = F.log_softmax(output, dim=1)
        args = [topic_vector, visual_value, visual_key,lang_value, lang_key,  kv_state]
        return logprobs, state, args

    def forward(self, feats, seq, seq_mask, dense_captions, dense_lengths):

        batch_size = feats.size(0)
        feat_dim = feats.size(2)
        sen_h, sen_c = self.init_sent_hidden(batch_size)
        pool_feats = self.feat_topic_embed(feats).max(dim=1)[0]
        predict_end = Variable(feats.data.new(batch_size, self.sent_max,2).zero_())
        predict_word = Variable(feats.data.new(batch_size, self.sent_max, seq.size(2) - 1,
                                          self.vocab_size-2).zero_())
        store_h = Variable(feats.data.new(batch_size, self.sent_max, self.sen_rnn_size).zero_())

        feats = feats.view(-1,feat_dim)
        att_feats = self.att_embed(feats).view(batch_size,-1,self.word_rnn_size)
        p_att_feats = self.ctx2att(att_feats)

        dense_hidden = self.embed(dense_captions)

        dense_hidden = dense_hidden.max(dim=2)[0]

        visual_key = p_att_feats
        visual_value = att_feats
        lang_key = self.lang_embed(dense_hidden)
        lang_value = dense_hidden
        kv_state = self.init_word_lstm2_hidden(batch_size)
        all_att_cost = []

        for i in range(self.sent_max):
            sen_h, sen_c = self.sen_lstm(pool_feats,(sen_h, sen_c))
            topic_vector = self.topic_layer(sen_h)
            logic_end = self.pre_prob(sen_h)
            predict_end[:, i,:] = logic_end
            store_h[:, i, :] = sen_h
            init_word_lstm2_hidden = self.init_word_lstm2_hidden(batch_size)
            state = (init_word_lstm2_hidden, init_word_lstm2_hidden)
            for j in range(seq.size(2)-1):
                if self.training and j >= 1 and self.ss_prob > 0.0:
                        sample_prob = feats.data.new(batch_size).uniform_(0, 1)
                        samp_mask = sample_prob < self.ss_prob
                        if samp_mask.sum() == 0:
                            it = seq[:, i, j ].clone()
                        else:
                            sample_ind = samp_mask.nonzero().view(-1)
                            it = seq[:, i, j].data.clone()
                            prob_prev = torch.exp(predict_word[:, i, j - 1].detach())
                            it.index_copy_(0, sample_ind,
                                           torch.multinomial(prob_prev, 1).view(-1).index_select(0, sample_ind) + 2)
                else:
                    it = seq[:, i, j].clone()
                if j >= 1 and seq[:, i, j].data.sum() == 0:
                    break
                xt = self.embed(it)

                output, visual_key, lang_key,att_cost, state,kv_state = self.word_lstm(xt,it, topic_vector, visual_value,
                  visual_key, lang_value, lang_key,self.kv_attention, self.updata_kv,self.max_att_step,state, kv_state)
                output = self.logit(output)
                output = F.log_softmax(output, dim=1)
                predict_word[:, i, j] = output
                all_att_cost.append(att_cost)
        return predict_word, predict_end, all_att_cost

    def sample_beam(self, feats, dense_captions, opt):
        beam_size = opt.get('beam_size', 10)
        stop_value = opt.get('stop_value', 0.5)
        batch_size = feats.size(0)
        feat_dim = feats.size(2)
        pool_feats = self.feat_topic_embed(feats).max(dim=1)[0]

        feats = feats.view(-1, feat_dim)
        att_feats = self.att_embed(feats).view(batch_size, -1, self.word_rnn_size)
        p_att_feats = self.ctx2att(att_feats)

        dense_hidden = self.embed(dense_captions)
        dense_hidden = dense_hidden.max(dim=2)[0]


        self.seq_length = self.word_max - 1
        seq = torch.LongTensor(batch_size,self.sent_max,self.seq_length, ).zero_()

        seqLogprobs = torch.FloatTensor(batch_size,self.sent_max, self.seq_length, )
        sent_end = torch.LongTensor(batch_size, self.sent_max).zero_()
        self.done_beams = [[[] for _ in range(self.sent_max)]  for _ in range(batch_size)]
        for k in range(batch_size):
            sen_h, sen_c = self.init_sent_hidden(beam_size)
            tmp_topic_feats = pool_feats[k:k + 1].expand(beam_size, pool_feats.size(1))
            tmp_att_feats = att_feats[k:k + 1].expand(*(beam_size,)+ att_feats.size()[1:])
            tmp_p_att_feats = p_att_feats[k:k + 1].expand(*(beam_size,)+ p_att_feats.size()[1:])

            tmp_dense_hidden = dense_hidden[k:k + 1].expand(*(beam_size,)+ dense_hidden.size()[1:])


            visual_key = tmp_p_att_feats
            visual_value = tmp_att_feats
            lang_key = self.lang_embed(tmp_dense_hidden)
            lang_value = tmp_dense_hidden
            kv_state = self.init_word_lstm2_hidden(beam_size)

            for s in range(self.sent_max):
                sen_h, sen_c = self.sen_lstm(tmp_topic_feats, (sen_h, sen_c))
                topic_vector = self.topic_layer(sen_h)
                logic_end = self.pre_prob(sen_h)  # .view(-1)
                logic_end = self.softmax(logic_end)
                init_word_lstm2_hidden = self.init_word_lstm2_hidden(beam_size)
                state = (init_word_lstm2_hidden, init_word_lstm2_hidden)
                if s == 0:
                    sent_end[k, s] = 1
                elif s >= 1:
                    if s == 1:
                        sent_unfinished = logic_end[:, 1] > stop_value
                    else:
                        sent_unfinished = sent_unfinished * (logic_end[:, 1] > stop_value)
                    sent_end[k, s] = sent_unfinished[-1]
                    if sent_unfinished.sum() == 0:
                        break
                # input <BOS>
                it = feats.data.new(beam_size).long().fill_(1)
                xt = self.embed(it)
                output, visual_key, lang_key, att_cost, state, kv_state = self.word_lstm(xt, it, topic_vector,
                                                                                         visual_value, visual_key,
                                                                                         lang_value, lang_key,
                                                                                         self.kv_attention,
                                                                                         self.updata_kv,
                                                                                         self.max_att_step, state,
                                                                                         kv_state)
                output = self.logit(output)
                logprobs = F.log_softmax(output, dim=1)
                if s>=1:
                   generated_seq = seq[k,:s,:]
                else:
                    generated_seq = feats.data.new(1, self.word_max-1).zero_()
                self.done_beams[k][s],visual_key,lang_key,kv_state = self.beam_search(state, logprobs, generated_seq,topic_vector,
                                                    visual_value, visual_key,lang_value, lang_key, kv_state, opt=opt)
                seq[k, s, :] = self.done_beams[k][s][0]['seq'] + 2  # the first beam has highest cumulative score
                seqLogprobs[k, s, :] = self.done_beams[k][s][0]['logps']
        last_sent_end = sent_end.unsqueeze(2)
        last_seq = seq * last_sent_end
        a = last_seq.detach().cpu().numpy()
        return last_seq,seqLogprobs


    def sample(self,feats,dense_captions, dense_lengths,opt={}):
        sample_max = opt.get('sample_max',1)
        beam_size = opt.get('beam_size',1)
        temperature = opt.get('tempeture',1.0)
        batch_size = feats.size(0)
        feat_dim = feats.size(2)
        stop_value = opt.get('stop_value',0.5)
        if beam_size > 1:
            return self.sample_beam(feats, dense_captions, opt)
        sen_h, sen_c = self.init_sent_hidden(batch_size)
        pool_feats = self.feat_topic_embed(feats).max(dim=1)[0]
        predict_seq = feats.data.new(batch_size, self.sent_max, self.word_max-1).long().zero_()
        seqLogprobs = feats.data.new(batch_size, self.sent_max, self.word_max-1).zero_()

        sent_unfinished = 1


        feats = feats.view(-1, feat_dim)
        att_feats = self.att_embed(feats).view(batch_size, -1, self.word_rnn_size)
        p_att_feats = self.ctx2att(att_feats)

        # language local phrases
        dense_hidden = self.embed(dense_captions)
        dense_hidden = dense_hidden.max(dim=2)[0]

        visual_key = p_att_feats
        visual_value = att_feats
        lang_key = self.lang_embed(dense_hidden)
        lang_value = dense_hidden
        kv_state = self.init_word_lstm2_hidden(batch_size)

        for i in range(self.sent_max):
            sen_h, sen_c = self.sen_lstm(pool_feats,(sen_h, sen_c))
            topic_vector = self.topic_layer(sen_h)
            logic_end = self.pre_prob(sen_h)
            logic_end = self.softmax(logic_end)
            init_word_lstm2_hidden = self.init_word_lstm2_hidden(batch_size)
            state = (init_word_lstm2_hidden, init_word_lstm2_hidden)
            if i >= 1:
                if i == 1:
                    sent_unfinished = logic_end[:, 1] > stop_value
                else:
                    sent_unfinished = sent_unfinished * (logic_end[:, 1] > stop_value)
                if sent_unfinished.sum() == 0:
                    break

            for j in range(self.word_max):

                if j == 0:
                    it = feats.data.new(batch_size).long().fill_(1)
                elif sample_max:
                    sampleLogprobs, it = torch.max(logprobs.data, 1)
                    it = it.view(-1).long()+2
                else:
                    if temperature == 1.0:
                        prob_prev = torch.exp(
                            logprobs.data)
                    else:
                        prob_prev = torch.exp(torch.div(logprobs.data, temperature))
                    it = torch.multinomial(prob_prev, 1)
                    sampleLogprobs = logprobs.gather(1, it)
                    it = it.view(-1).long() + 2

                xt = self.embed(it)

                if j >= 1:
                    if j == 1:
                        word_unifished = (it != 2) * sent_unfinished
                    else:
                        word_unifished = word_unifished * (it != 2)
                    if word_unifished.sum() == 0:
                        break
                    it = it * word_unifished.type_as(it)
                    predict_seq[:, i, j - 1] = it
                    seqLogprobs[:, i, j - 1] = sampleLogprobs.view(-1)
                output, visual_key, lang_key, att_cost, state, kv_state = self.word_lstm(xt, it, topic_vector,
                                                                                         visual_value, visual_key,
                                                                                         lang_value, lang_key,
                                                                                         self.kv_attention,
                                                                                         self.updata_kv,
                                                                                         self.max_att_step, state,
                                                                                         kv_state)
                output = self.logit(output)
                logprobs = F.log_softmax(output, dim=1)

        return predict_seq,seqLogprobs  # word from <PAD> 0
