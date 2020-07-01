from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import time
import misc.utils as utils
from collections import OrderedDict
import torch
from torch.autograd import Variable
import sys
sys.path.append('coco_caption')
#from coco_caption.pyciderevalcap.ciderD.ciderD import CiderD

from cider.pyciderevalcap.ciderD.ciderD import CiderD
CiderD_scorer = CiderD(df='para-train-idxs')
#CiderD_scorer = CiderD(df='corpus')


def array_to_str(arr,add_eos=True):  # arr:[6,max_len]
    out = ''


    for i in range(len(arr)):
        sent = arr[i]
        if sent.sum() == 0:
            break
        for j in range(len(sent)):
            if sent[j] == 0:
                if add_eos:
                   out = out + '2 '
                break
            else:
                out += str(sent[j]) + ' '
    return out.strip()


def get_self_critical_reward(model, feats,dense_labels,dense_lengths, data, gen_result):


    batch_size = gen_result.size(0)
    model.eval()
    with torch.no_grad():

         greedy_res, _ = model.sample(feats, dense_labels, dense_lengths)
    model.train()
    res = OrderedDict()
    
    gen_result = gen_result.cpu().numpy()
    greedy_res = greedy_res.cpu().numpy()
    for i in range(batch_size):
        res[i] = [array_to_str(gen_result[i])]
    for i in range(batch_size):
        res[batch_size + i] = [array_to_str(greedy_res[i])]

    gts = OrderedDict()
    for i in range(len(data['gts'])):

        gts[i] = [array_to_str(data['gts'][i],False)]


    res = [{'image_id':i, 'caption': res[i]} for i in range(2 * batch_size)]
    gts = {i: gts[i % batch_size] for i in range(2 * batch_size)}
    _, scores = CiderD_scorer.compute_score(gts, res)

    scores = scores[:batch_size] - scores[batch_size:]

    rewards = np.repeat(scores[:, np.newaxis], gen_result.shape[1], 1)
    rewards = np.repeat(rewards[:,:, np.newaxis], gen_result.shape[2], 2)
    return rewards
