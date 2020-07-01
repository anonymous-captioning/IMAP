from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np
import json
from json import encoder
import random
import os
import sys
import misc.utils as utils
import torch.nn.functional as F


# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def language_eval(dataset, preds,  split):
    sys.path.append("coco_caption")
    annFile = 'coco_caption/annotations/para_captions_'+split+'.json'
    from coco_caption.pycocotools.coco import COCO
    from coco_caption.pycocoevalcap.eval import COCOEvalCap

    encoder.FLOAT_REPR = lambda o: format(o, '.3f')

    if not os.path.isdir('eval_results'):
        os.mkdir('eval_results')
    cache_path = os.path.join('eval_results/',  '_' + split + '.json')

    coco = COCO(annFile)
    valids = coco.getImgIds()

    # filter results to only those in MSCOCO validation set (will be about a third)
    preds_filt = [p for p in preds if p['image_id'] in valids]
    print('using %d/%d predictions' % (len(preds_filt), len(preds)))
    json.dump(preds_filt, open(cache_path, 'w'))  # serialize to temporary json file. Sigh, COCO API...

    cocoRes = coco.loadRes(cache_path)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    cocoEval.evaluate()

    # create output dictionary
    out = {}
    for metric, score in cocoEval.eval.items():
        out[metric] = score

    imgToEval = cocoEval.imgToEval
    for p in preds_filt:
        image_id, caption = p['image_id'], p['caption']
        imgToEval[image_id]['caption'] = caption
    with open(cache_path, 'w') as outfile:
        json.dump({'overall': out, 'imgToEval': imgToEval}, outfile)

    return out

def eval_split(model,crit, loader, eval_kwags={}):
    verbose = eval_kwags.get('verbose',False)
    num_images = eval_kwags.get('num_images', eval_kwags.get('val_images_use',-1))
    split = eval_kwags.get('split','val')
    lang_eval = eval_kwags.get('language_eval',0)
    dataset = eval_kwags.get('dataset','coco')
    beam_size = eval_kwags.get('beam_size',1)

    model.eval()
    loader.reset_iterator(split)

    n = 0
    loss_word_sum = 0
    loss_end_sum = 0
    loss_sum = 0
    loss_evals = 1e-8
    predictions = []
    while True:
        data = loader.get_batch(split)
        n = n + loader.batch_size
        tmp = [data['att_feats'], data['labels'], data['masks'], data['sent_place'], data['dense_captions'],
               data['dense_lengths'], ]
        tmp = [Variable(torch.from_numpy(_), requires_grad=False).cuda() for _ in tmp]
        feats, labels, masks, sent_place, dense_captions, dense_lengths = tmp
        # with torch.no_grad():
        #      predict_word, predict_end = model(feats, labels, masks,dense_captions, dense_lengths)
        #      loss_word, loss_end, loss = crit(predict_word, predict_end, labels[:, :, 1:], sent_place, masks[:, :, 1:],
        #                                  loader)
        # loss_word_sum = loss_word_sum+loss_word
        # loss_end_sum = loss_end_sum+loss_end
        # loss_sum = loss_sum+loss
        loss_evals = loss_evals + 1

        seq,_ = model.sample(feats,dense_captions, dense_lengths,opt=eval_kwags)  # [batch,sen_max,word_max]
        sents = utils.decode_sequence(loader.get_vocab(), seq.data)
        for k, sent in enumerate(sents):

            entry = {'image_id': data['infos'][k]['id'], 'caption': sent}
            predictions.append(entry)

            # if verbose and random.random() < 0.01:
            print('image %s: %s' % (entry['image_id'], entry['caption']))


        # if we wrapped around the split or used up val imgs budget then bail
        ix0 = data['bounds']['it_pos_now']
        ix1 = data['bounds']['it_max']
        if num_images != -1:
            ix1 = min(ix1, num_images)
        for i in range(n - ix1):
            predictions.pop()
        json.dump(predictions,open('val_result.json','w'))
        if verbose and ix0 % 1 == 0:
            print('evaluating validation preformance... %d/%d ' % (ix0 - 1, ix1))

        if data['bounds']['wrapped']:
            break
        if num_images >= 0 and n >= num_images:
            break

    lang_stats = None
    if lang_eval == 1:
        lang_stats = language_eval(dataset, predictions,  split)

    # Switch back to training mode
    model.train()
    # return   loss_word/loss_evals, loss_end/loss_evals, loss/loss_evals,predictions, lang_stats
    return   1e-8/loss_evals, 1e-8/loss_evals, 1e-8/loss_evals,predictions, lang_stats