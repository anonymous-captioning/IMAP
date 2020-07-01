from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import argparse
from random import shuffle, seed
import string

import h5py
import numpy as np
import torch
# import torchvision.models as models
import skimage.io
from PIL import Image


def build_vocab(imgs, params):
    count_thr = params['word_count_threshold']
    # count up the number of words
    counts = {}
    for img in imgs:
        sents = img['sentences']
        for sent in sents['tokens']:
            sent =sent.replace(',',' , ').split(' ')
            for w in sent:
                if w != '' and w != ' ':
                    counts[w] = counts.get(w, 0) + 1
    cw = sorted([(count, w) for w, count in counts.items()], reverse=True)
    print('top words and their counts:')
    print('\n'.join(map(str, cw[:20])))

    # print some stats
    total_words = sum(counts.values())
    print('total words:', total_words)
    bad_words = [w for w, n in counts.items() if n <= count_thr ]
    vocab = [w for w, n in counts.items() if n > count_thr]
    bad_count = sum(counts[w] for w in bad_words)
    print('number of bad words: %d/%d = %.2f%%' % (len(bad_words), len(counts), len(bad_words) * 100.0 / len(counts)))
    print('number of words in vocab would be %d' % (len(vocab),))
    print('number of UNKs: %d/%d = %.2f%%' % (bad_count, total_words, bad_count * 100.0 / total_words))

    # lets look at the distribution of lengths as well
    sent_lengths = {}

    for img in imgs:
        sent = img['sentences']
        for txt in  sent['tokens']:
            txt = txt.replace(',', ' , ').split(' ')
            nw = len(txt)
            if nw == 0 or nw == 1 or nw == 2:
                continue
            sent_lengths[nw] = sent_lengths.get(nw, 0) + 1
    max_len = max(sent_lengths.keys())
    print('max length sentence in raw data: ', max_len)
    print('sentence length distribution (count, number of words):')
    sum_len = sum(sent_lengths.values())

    top_30 = sum(list(sent_lengths.values())[:30])  # top30:99%
    print('top 30 sentence length : %f', top_30 * 100.0 / sum_len)
    for i in range(max_len + 1):
        print('%2d: %10d   %f%%' % (i, sent_lengths.get(i, 0), sent_lengths.get(i, 0) * 100.0 / sum_len))

    ix2word = {}
    ix2word[0] = '<PAD>'
    ix2word[1] = '<BOS>'
    ix2word[2] = '<EOS>'
    ix2word[3] = '<UNK>'

    word2ix = {}
    word2ix['<PAD>'] = 0
    word2ix['<BOS>'] = 1
    word2ix['<EOS>'] = 2
    word2ix['<UNK>'] = 3

    for idx, w in enumerate(vocab):
        word2ix[w] = idx + 4
        ix2word[idx + 4] = w

    for img in imgs:
        img['final_captions'] = []
        sents = img['sentences']
        for sent in  sents['tokens']:
            sent = '<BOS> ' + sent + ' <EOS>'
            sent = sent.split(' ')
            if len(sent) < 5:
                continue
            caption = [w if w in word2ix else '<UNK>' for w in sent]

            img['final_captions'].append(caption)

        # process the dense captions
        img['fianl_dense_captions'] = []
        denses = img['dense_captions']
        for dense in denses:
            dense = dense.split(' ')
            den_cap = [w if w in word2ix else '<UNK>' for w in dense]
            img['fianl_dense_captions'].append(den_cap)

    return ix2word, word2ix


def encode_captions(imgs, params, wtoi):
    """
    encode all captions into one large array, which will be 1-indexed.
    also produces label_start_ix and label_end_ix which store 1-indexed
    and inclusive (Lua-style) pointers to the first and last caption for
    each image in the dataset.
    """

    max_length = params['max_length']
    N = len(imgs)
    M = sum(len(img['final_captions']) for img in imgs)  # total number of captions

    label_arrays = []
    label_start_ix = np.zeros(N, dtype='uint32')  # note: these will be one-indexed
    label_end_ix = np.zeros(N, dtype='uint32')
    counter = 1
    for i, img in enumerate(imgs):
        n = len(img['final_captions'])
        assert n > 0, 'error: some image has no captions'

        Li = np.zeros((n, max_length), dtype='uint32')  # '<PAD>'
        for j, s in enumerate(img['final_captions']):
            for k, w in enumerate(s):
                if k < max_length:
                    Li[j, k] = wtoi[w]

        label_arrays.append(Li)
        label_start_ix[i] = counter
        label_end_ix[i] = counter + n - 1

        counter += n

    L = np.concatenate(label_arrays, axis=0)
    assert L.shape[0] == M, 'lengths don\'t match? that\'s weird'

    print('encoded captions to array of size ', L.shape)

    # process the dense captions
    max_dense_length = params['max_dense_length']
    dense_arrays = []
    dense_lengths = []
    cnt_dense = 0
    for img in imgs:
        n = len(img['fianl_dense_captions'])
        Di = np.zeros((n, max_dense_length), dtype='uint32')  # '<PAD>'
        dense_length = []
        for j, s in enumerate(img['fianl_dense_captions']):
            if len(s) > max_dense_length:
                cnt_dense = cnt_dense + 1
            dense_length.append(min(len(s), max_dense_length))
            for k, w in enumerate(s):
                if k < max_dense_length:
                    Di[j, k] = wtoi[w]
        dense_lengths.append(dense_length)
        dense_arrays.append(Di)
    print('there are ', cnt_dense, ' dense captions > max_dense_length')
    D = np.array(dense_arrays)
    D_L = np.array(dense_lengths, dtype='float32')
    assert D.shape[0] == D_L.shape[0] == N, 'lengths don\'t match? that\'s weird'
    return L, label_start_ix, label_end_ix, D, D_L

def main(params):
    imgs = json.load(open(params['input_json'], 'r'))
    imgs = imgs['images']
    seed(123)


    ix2word, word2ix = build_vocab(imgs, params)

    # encode captions in large arrays, ready to ship to hdf5 file
    L, label_start_ix, label_end_ix,D,D_L = encode_captions(imgs, params, word2ix)

    # create output h5 file
    N = len(imgs)
    f_lb = h5py.File(params['output_h5'] + '_label.h5', "w")
    f_lb.create_dataset("labels", dtype='uint32', data=L)
    f_lb.create_dataset("label_start_ix", dtype='uint32', data=label_start_ix)
    f_lb.create_dataset("label_end_ix", dtype='uint32', data=label_end_ix)
    f_lb.create_dataset("denses", dtype='uint32', data=D)
    f_lb.create_dataset("dense_lengths", dtype='float32', data=D_L)

    f_lb.close()

    # create output json file
    out = {}
    out['ix_to_word'] = ix2word  # encode the (1-indexed) vocab
    out['images'] = []
    for i, img in enumerate(imgs):

        jimg = {}
        jimg['split'] = img['split']

        if 'id' in img: jimg['id'] = img['id']
        out['images'].append(jimg)

    json.dump(out, open(params['output_json'], 'w'))
    print('wrote ', params['output_json'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # input json
    parser.add_argument('--input_json', default='../data/captions/para_paragraph_caption_dense.json',
                        help='input json file to process into hdf5')

    parser.add_argument('--output_json', default='../data/paratalk.json', help='output json file')
    parser.add_argument('--output_h5', default='../data/paratalk', help='output h5 file')
    parser.add_argument('--images_root', default='',
                        help='root location in which images are stored, to be prepended to file_path in input json')

    # options
    parser.add_argument('--max_length', default=30, type=int,
                        help='max length of sentence')
    parser.add_argument('--max_dense_length', default=8, type=int,
                        help='max num of each dense captions')
    parser.add_argument('--word_count_threshold', default=5, type=int,
                        help='only words that occur more than this number of times will be put in vocab')

    args = parser.parse_args()
    params = vars(args)  # convert to ordinary dict
    print('parsed input parameters:')
    print(json.dumps(params, indent=2))
    main(params)
