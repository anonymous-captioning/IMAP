from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import h5py
import os
import numpy as np
import random
import pickle as cPickle
import torch.utils.data as data
import torch
import multiprocessing
from misc.utils import decode_sequence

def get_npy_data(ix, fc_file, att_file, use_att):
    if use_att == True:
        return (np.load(fc_file), np.load(att_file)['feat'], ix)
    else:
        return (np.load(fc_file), np.zeros((1, 1, 1)), ix)


class DataLoader(data.Dataset):
    def reset_iterator(self, split):
        del self._prefetch_process[split]
        self._prefetch_process[split] = BlobFetcher(split, self, split == 'train')
        self.iterators[split] = 0

    def get_vocab_size(self):
        return self.vocab_size

    def get_vocab(self):
        return self.ix_to_word

    def get_seq_length(self):
        return self.seq_length

    def __init__(self, opt):
        self.opt = opt
        self.batch_size = self.opt.batch_size
        self.use_att = getattr(opt, 'use_att', True)
        self.sent_max = opt.sen_max
        self.word_max = opt.word_max
        print('DataLoader loading json file: ', opt.input_json)
        self.info = json.load(open(self.opt.input_json))
        self.ix_to_word = self.info['ix_to_word']

        self.vocab_size = len(self.ix_to_word)  # 4158
        print('vocab size is ', self.vocab_size)


        print('DataLoader loading h5 file: ', opt.input_att_dir, opt.input_label_h5)
        self.h5_label_file = h5py.File(self.opt.input_label_h5, 'r', driver='core')

        self.input_att_dir = self.opt.input_att_dir
        self.att_feats = h5py.File(self.input_att_dir,'r')
        # load in the sequence data
        seq_size = self.h5_label_file['labels'].shape
        self.seq_length = seq_size[1]   # 30
        self.per_box = opt.per_box
        print('max sequence length in data is', self.seq_length)
        self.label_start_ix = self.h5_label_file['label_start_ix'][:]
        self.label_end_ix = self.h5_label_file['label_end_ix'][:]
        self.dense_captions = self.h5_label_file['denses'][:]
        self.dense_lengths = self.h5_label_file['dense_lengths'][:]

        self.num_images = self.label_start_ix.shape[0]
        print('read %d image features' % (self.num_images))

        self.split_ix = {'train': [], 'val': [], 'test': []}
        for ix in range(len(self.info['images'])):
            img = self.info['images'][ix]
            if img['split'] == 'train':
                self.split_ix['train'].append(ix)
            elif img['split'] == 'val':
                self.split_ix['val'].append(ix)
            elif img['split'] == 'test':
                self.split_ix['test'].append(ix)


        print('assigned %d images to split train' % len(self.split_ix['train']))
        print('assigned %d images to split val' % len(self.split_ix['val']))
        print('assigned %d images to split test' % len(self.split_ix['test']))

        self.iterators = {'train': 0, 'val': 0, 'test': 0}

        self._prefetch_process = {}  # The three prefetch process
        for split in self.iterators.keys():
            self._prefetch_process[split] = BlobFetcher(split, self, split == 'train')

        # Terminate the child process when the parent exists
        def cleanup():
            print('Terminating BlobFetcher')
            for split in self.iterators.keys():
                del self._prefetch_process[split]

        import atexit
        atexit.register(cleanup)
    # NOTE: Not use fc feats
    def get_batch(self, split, batch_size=None, seq_per_img=None):
        batch_size = batch_size or self.batch_size

        att_batch = []

        imag_cap_distribution = np.zeros([batch_size, self.sent_max], 'int')
        label_batch = np.zeros([batch_size * self.sent_max, self.word_max], dtype='int')
        mask_batch = np.zeros([batch_size * self.sent_max, self.word_max], dtype = 'float32')

        wrapped = False
        infos, gts, denses_label, dense_lengths = [], [], [], []
        for i in range(batch_size):
            tmp_att, ix, tmp_wrapped = self._prefetch_process[split].get()
            att_batch.append(tmp_att)
            denses_label.append(self.dense_captions[ix])
            dense_lengths.append(self.dense_lengths[ix])

            ix1 = self.label_start_ix[ix] - 1
            ix2 = self.label_end_ix[ix] - 1
            ncap = ix2 - ix1 + 1
            imag_cap_distribution[i,:ncap] = 1
            assert ncap > 0, 'an image does not have any label. this can be handled but right now isn\'t'

            if ncap < self.sent_max:
                seq = np.zeros([self.sent_max, self.seq_length], dtype='int')
                for q in range(ncap):
                    seq[q, :] = self.h5_label_file['labels'][ix1+q, :self.seq_length]
            else:
                seq = self.h5_label_file['labels'][ix1: ix1 + self.sent_max, :self.seq_length]

            label_batch[i * self.sent_max: (i + 1) * self.sent_max, :] = seq


            if tmp_wrapped:
                wrapped = True


            gts.append(self.h5_label_file['labels'][self.label_start_ix[ix] - 1: self.label_end_ix[ix],1:])

            # record associated info as well
            info_dict = {}
            info_dict['ix'] = ix
            info_dict['id'] = self.info['images'][ix]['id']
            infos.append(info_dict)

        nonzeros = np.array(list(map(lambda x: (x != 0).sum(), label_batch)))
        for ix, row in enumerate(mask_batch):
            row[:nonzeros[ix]] = 1

        mask_batch = mask_batch.reshape(batch_size, -1, self.word_max)
        data = {}
        data['att_feats'] = np.stack(att_batch)
        data['labels'] = label_batch.reshape(batch_size, -1, self.word_max)
        data['gts'] = np.array(gts)
        data['masks'] = mask_batch
        data['bounds'] = {'it_pos_now': self.iterators[split], 'it_max': len(self.split_ix[split]), 'wrapped': wrapped}
        data['infos'] = infos
        data['sent_place'] = imag_cap_distribution
        data['dense_captions'] = np.array(denses_label, dtype='int')
        data['dense_lengths'] = np.array(dense_lengths, dtype='float32')

        return data

    # It's not coherent to make DataLoader a subclass of Dataset, but essentially, we only need to implement the following to functions,
    # so that the torch.utils.data.DataLoader can load the data according the index.
    # However, it's minimum change to switch to pytorch data loading.
    def __getitem__(self, index):
        """This function returns a tuple that is further passed to collate_fn
        """
        ix = index  # self.split_ix[index]
        att_feat =  np.array(self.att_feats['feats'][str(self.info['images'][ix]['id'])])
        #att_feat = np.load(os.path.join(self.input_att_dir, str(self.info['images'][ix]['id']) + '.npz'))['feat']
        #fc_feat = np.load(os.path.join(self.input_fc_dir, str(self.info['images'][ix]['id']) + '.npy'))
        return att_feat, ix

    def __len__(self):
        return len(self.info['images'])


class SubsetSampler(torch.utils.data.sampler.Sampler):
    r"""Samples elements randomly from a given list of indices, without replacement.
    Arguments:
        indices (list): a list of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)


class BlobFetcher():
    """Experimental class for prefetching blobs in a separate process."""

    def __init__(self, split, dataloader, if_shuffle=False):
        """
        db is a list of tuples containing: imcrop_name, caption, bbox_feat of gt box, imname
        """
        self.split = split
        self.dataloader = dataloader
        self.if_shuffle = if_shuffle

    # Add more in the queue
    def reset(self):
        """
        Two cases:
        1. not hasattr(self, 'split_loader'): Resume from previous training. Create the dataset given the saved split_ix and iterator
        2. wrapped: a new epoch, the split_ix and iterator have been updated in the get_minibatch_inds already.
        """
        # batch_size is 0, the merge is done in DataLoader class
        self.split_loader = iter(data.DataLoader(dataset=self.dataloader,
                                                 batch_size=1,
                                                 sampler=SubsetSampler(self.dataloader.split_ix[self.split][
                                                                       self.dataloader.iterators[self.split]:]),
                                                 shuffle=False,
                                                 pin_memory=True,
                                                 num_workers= 1,  # multiprocessing.cpu_count(),
                                                 # HDF5 file doesnot support multiprocess read
                                                 collate_fn=lambda x: x[0]))

    def _get_next_minibatch_inds(self):
        max_index = len(self.dataloader.split_ix[self.split])
        wrapped = False  # check if a new epoch starts

        ri = self.dataloader.iterators[self.split]  # reindex
        ix = self.dataloader.split_ix[self.split][ri]

        ri_next = ri + 1
        if ri_next >= max_index:
            ri_next = 0
            if self.if_shuffle:
                random.shuffle(self.dataloader.split_ix[self.split])
            wrapped = True
        self.dataloader.iterators[self.split] = ri_next

        return ix, wrapped

    def get(self):
        if not hasattr(self, 'split_loader'):
            self.reset()

        ix, wrapped = self._get_next_minibatch_inds()
        tmp = self.split_loader.next()
        if wrapped:
            self.reset()

        assert tmp[1] == ix, "ix not equal"

        return tmp + [wrapped]
