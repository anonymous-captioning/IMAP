import torch
import torch.nn as nn
from torch.autograd import *
import json
import numpy as np
from collections import OrderedDict
import h5py

# <PAD>=0  <BOS>=1  <EOS>=2  <UNK>=3
class dense_caption(nn.Module):
    def __init__(self,paragraph_word):
        super(dense_caption,self).__init__()
        self.paragraph_i2w = paragraph_word
        self.parag_vocab_size = len(self.paragraph_i2w)
        self.feats_size = 4096
        self.word_encoding_size = 512
        self.rnn_size = 512
        self.vocab_size = len(self.paragraph_i2w)  # 4158
        # self.START_TOKEN = self.vocab_size + 1
        # self.END_TOKEN = self.vocab_size + 1
        # self.NULL_TOKEN = self.vocab_size + 2

        self.feats_embed = nn.Linear(self.feats_size,self.word_encoding_size)
        self.feats_trans = nn.Sequential(*(self.feats_embed,nn.ReLU()))
        self.lstm = nn.LSTMCell(self.word_encoding_size,self.rnn_size)
        self.word_embed = nn.Embedding(self.vocab_size, self.word_encoding_size)
        # self.word_linear = nn.Linear(self.rnn_size,self.vocab_size+1)
        self.word_linear = nn.Linear(self.rnn_size,self.vocab_size-2)

        self.seq_length = 15
        last = json.load(open('dense_exact.json'))
        self.word_vector_file = last['dense_word_vector']
        self.input_json = last['dense_model_opt']
        self.rnn_file = last['dense_rnn']
        self.embedding_file = last['dense_img_word_linear']
        self.init_from_dense()
    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (Variable(weight.new(bsz, self.rnn_size).zero_()),
                Variable(weight.new(bsz, self.rnn_size).zero_()))
    def init_lstm_from_dence(self,weight_bias):
        weight,bias = np.array(weight_bias['weight'],dtype='float32'),np.array(weight_bias['bias'],dtype='float32')

        weight_i,weight_f,weight_o,weight_g = weight[:,:512],weight[:,512:2*512],\
                                              weight[:,512*2:512*3],weight[:,512*3:]

        bias_i, bias_f, bias_o, bias_g = bias[:512], bias[512:2 * 512],bias[512 * 2:512 * 3], bias[512 * 3:]
        tmp_weight = [weight_i,weight_f,weight_g,weight_o]
        tmp_bias =   [bias_i, bias_f, bias_g, bias_o]
        tmp_w = [torch.FloatTensor(_) for _ in tmp_weight]  # [4,1024,512]
        tmp_b = [torch.FloatTensor(_) for _ in tmp_bias]
        load_weight = torch.cat(tmp_w,dim=1).squeeze().permute(1,0) # # [1024,2048]---[2048,1024]
        load_b = torch.cat(tmp_b,dim=0).squeeze()  # [4,512]---[2048]
        self.lstm.weight_ih = torch.nn.Parameter(load_weight[:,:512])
        self.lstm.weight_hh = torch.nn.Parameter(load_weight[:, 512:])
        self.lstm.bias_ih,self.lstm.bias_hh = torch.nn.Parameter(load_b*0.5),torch.nn.Parameter(load_b*0.5)

    def _make_emb_state_dict(self, words, word_vectors):
        weight = torch.zeros(len(words), self.word_encoding_size)
        for idx, word in words.items():
            weight[int(idx),:] = torch.from_numpy(np.array(word_vectors[word], dtype='float32'))
        state_dict = OrderedDict({'weight': weight})
        return state_dict


    def init_from_dense(self):
        word_vectors = self.word_vector_file
        info = self.input_json
        weight_bias = self.rnn_file
        # init lstm
        self.init_lstm_from_dence(weight_bias)

        # init word linear and img embedding
        img_word_linear = self.embedding_file
        tmp = [img_word_linear['img_weight'],img_word_linear['bias'],
               img_word_linear['word_linear_weight'],img_word_linear['word_linear_bias']]
        process = [torch.from_numpy(np.array(_,dtype='float32')) for _ in tmp]
        self.feats_embed.weight = torch.nn.Parameter(process[0])
        self.feats_embed.bias = torch.nn.Parameter(process[1])

        state_dict = self._make_emb_state_dict(self.paragraph_i2w, word_vectors)
        self.word_embed.load_state_dict(state_dict)

        self.ix2word = info['idx_to_token']
        vgw2i = {j:i for i,j in enumerate(self.ix2word)}
        self.idx = []
        self.idx.append(10509)
        self.last_out_word = list(self.paragraph_i2w.values())[2:]  # start from <EOS>
        for word in self.last_out_word[1:]:  #  start from <UNK>
            if word in self.ix2word:
                self.idx.append(vgw2i[word])
        self.word_linear.weight = torch.nn.Parameter(process[2][self.idx])
        self.word_linear.bias = torch.nn.Parameter(process[3][self.idx])
    def decode(self,seq, ix2word):
        batch, seq_len = seq.size(0), seq.size(1)
        captions = []
        for i in range(batch):
            caption = ''
            for j,value in enumerate(seq[i,:].data.cpu().numpy()):
                value = value
                if value == 0:
                   break
                else:
                    if j==0:
                       caption = caption+ix2word[value]
                    else:
                        caption = caption+ ' ' + ix2word[value]
            captions.append(caption)
        return captions

    def forward(self,img_features):  # img_features:[50,4096]
        batch_size = img_features.size(0)
        img_vectors = self.feats_trans(img_features)  # [50,512]
        img_vectors = img_vectors  # [50,512]

        predict_word = Variable(img_vectors.data.new(batch_size, self.seq_length).long().zero_(),requires_grad=False)

        # init lstm state with img_vectors
        init_h,init_c = self.init_hidden(batch_size)
        h,c = self.lstm(img_vectors,(init_h,init_c))
        for i in range(self.seq_length):
            if i ==0:
                words = torch.cuda.LongTensor(batch_size).fill_(1)
            else:
                words = predict_word[:,i-1]+2
            input = self.word_embed(words)  # [batch,512]
            h,c = self.lstm(input,(h,c))
            word_prob = self.word_linear(h)
            _,idx = torch.max(word_prob,1)
            predict_word[:,i] = idx
        out = self.decode(predict_word,self.last_out_word)
        return out

paragraph_word = json.load(open('paratalk.json'))['ix_to_word']
input_json = json.load(open('data/para_paragraph_caption.json'))
model = dense_caption(paragraph_word).cuda()
test_file = 'VG_feature.h5'
feats = h5py.File(test_file,'r')['feats']
imgs = input_json['images']
for cnt,img in enumerate(imgs):
    print('\rnow propocess images :{}/{}'.format(cnt, len(imgs)), end='')
    img_id = img['id']
    img_feats = torch.from_numpy(np.array(feats[str(img_id)], dtype='float32')).cuda()
    out = model(img_feats)
    img['dense_captions'] = out

json.dump({'images':imgs}, open('data/para_paragraph_caption_dense.json', 'w'))