import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
def to_contiguous(tensor):
    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()

class RewardCriterion(nn.Module):
    def __init__(self):
        super(RewardCriterion, self).__init__()

    def forward(self, input, seq, reward):
        input = to_contiguous(input).view(-1)
        reward = to_contiguous(reward).view(-1)
        mask = (seq>0).float()
        mask = to_contiguous(torch.cat([mask.new(mask.size(0),mask.size(1),1).fill_(1), mask[:,:, :-1]], 2)).view(-1)
        output = - input * reward * mask
        output = torch.sum(output) / torch.sum(mask)

        return output


class Criterion(nn.Module):
    def __init__(self,opt):
        super(Criterion,self).__init__()
        self.weight_sen = opt.weight_sen
        self.weight_word = opt.weight_word

        self.loss = nn.CrossEntropyLoss(size_average=False)
        self.aat_lambda = opt.aat_lambda
    def forward(self, pred_seq,pred_end, target_seq,  target_end,mask,all_att_cost,loader):


        batch_size = pred_seq.size(0)
        target_end = target_end[:,:pred_end.size(1)].view(-1)
        pred_end = pred_end.view(-1,2)
        additional_add = np.array(mask==0,dtype='int')*2
        additional_add = torch.from_numpy(additional_add).cuda()

        target_seq = target_seq[:,:,:pred_seq.size(2)]+additional_add-2
        mask = mask[:,:,:pred_seq.size(2)]

        pred_seq = to_contiguous(pred_seq).view(-1,pred_seq.size(3))
        target_seq = to_contiguous(target_seq).view(-1,1)
        mask = to_contiguous(mask).view(-1,1)
        output = -pred_seq.gather(1,target_seq) * mask
        loss_seq = torch.sum(output) / batch_size

        loss_sen = self.loss(pred_end, target_end) / batch_size

        loss_att = torch.stack(all_att_cost).sum() / batch_size

        loss_all = loss_seq*self.weight_word + loss_sen*self.weight_sen + self.aat_lambda*loss_att
        return loss_seq,loss_sen,loss_att,loss_all



def set_lr(optimizer,lr):
    for group in optimizer.param_groups:
        group['lr'] = lr


def clip_gradient(optimizer,grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None and param.requires_grad:
                param.grad.data.clamp_(-grad_clip,grad_clip)

def input_data(data):
    tmp = [data['att_feats'], data['labels'], data['masks'], data['sent_place'],data['dense_captions'],data['dense_lengths']]
    tmp = [Variable(torch.from_numpy(_), requires_grad=False).cuda() for _ in tmp]
    feats, labels, masks, sent_place, dense_captions,dense_lengths = tmp
    batch_size = feats.size(0)

    return feats, labels, masks, sent_place, dense_captions,dense_lengths

def decode_sequence(ix_to_word, seq, add_vocab_size=0):

    N = len(seq)
    out = []
    for i in range(N):
        sents = ''
        for sentence in seq[i]:
            sent = ''
            if sentence.sum() ==0:  # all zeros
                break
            for j in range(len(sentence)):
                ix = sentence[j] -add_vocab_size
                if ix == 2:
                    sent = sent + ' . '
                    break
                elif ix > 2:
                    if j >=1:
                        sent = sent + ' '
                    sent = sent + ix_to_word[str(ix.item())]
            if ' . ' not in sent:
                sent = sent + ' . '
            sents = sents + sent
        out.append(sents)
    return out

def penalty_builder(penalty_config):
    if penalty_config == '':
        return lambda x,y: y
    pen_type, alpha = penalty_config.split('_')
    alpha = float(alpha)
    if pen_type == 'wu':
        return lambda x,y: length_wu(x,y,alpha)
    if pen_type == 'avg':
        return lambda x,y: length_average(x,y,alpha)

def length_wu(length, logprobs, alpha=0.):
    """
    NMT length re-ranking score from
    "Google's Neural Machine Translation System" :cite:`wu2016google`.
    """

    modifier = (((5 + length) ** alpha) /
                ((5 + 1) ** alpha))
    return (logprobs / modifier)

def length_average(length, logprobs, alpha=0.):
    """
    Returns the average probability of tokens in a sequence.
    """
    return logprobs / length