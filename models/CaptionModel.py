from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *
import misc.utils as utils
from functools import reduce

class CaptionModel(nn.Module):
    def __init__(self):
        super(CaptionModel, self).__init__()


    def beam_search(self, init_state, init_logprobs,generated_seq, *args, **kwargs):

        def add_diversity(beam_seq_table, logprobsf, t, divm, diversity_lambda, bdash):
            local_time = t - divm
            unaug_logprobsf = logprobsf.clone()
            for prev_choice in range(divm):
                prev_decisions = beam_seq_table[prev_choice][local_time]
                for sub_beam in range(bdash):
                    for prev_labels in range(bdash):
                        logprobsf[sub_beam][prev_decisions[prev_labels]] = logprobsf[sub_beam][prev_decisions[
                            prev_labels]] - diversity_lambda
            return unaug_logprobsf


        def beam_step(logprobsf, unaug_logprobsf, beam_size, t, beam_seq, beam_seq_logprobs,
                      beam_logprobs_sum, state, visual_key,lang_key, kv_state):

            ys, ix = torch.sort(logprobsf, 1, True)
            candidates = []
            cols = min(beam_size, ys.size(1))
            rows = beam_size
            if t == 0:
                rows = 1
            for c in range(cols):
                for q in range(rows):
                    local_logprob = ys[q, c].item()
                    candidate_logprob = beam_logprobs_sum[q] + local_logprob
                    local_unaug_logprob = unaug_logprobsf[q, ix[q, c]]
                    candidates.append({'c': ix[q, c], 'q': q, 'p': candidate_logprob, 'r': local_unaug_logprob})
            candidates = sorted(candidates, key=lambda x: -x['p'])
            a,b = candidates[0]['c'].item(),candidates[0]['r'].item()
            new_state = [_.clone() for _ in state]

            new_visual_key = visual_key.clone()
            new_lang_key = lang_key.clone()
            new_kv_state = kv_state.clone()
            if t >= 1:
                beam_seq_prev = beam_seq[:t].clone()
                beam_seq_logprobs_prev = beam_seq_logprobs[:t].clone()
            for vix in range(beam_size):
                v = candidates[vix]
                if t >= 1:
                    beam_seq[:t, vix] = beam_seq_prev[:, v['q']]
                    beam_seq_logprobs[:t, vix] = beam_seq_logprobs_prev[:, v['q']]
                for state_ix in range(len(new_state)):
                    new_state[state_ix][:, vix] = state[state_ix][:, v['q']]

                new_visual_key[vix] = visual_key[v['q']]
                new_lang_key[vix] = lang_key[v['q']]
                for kv_state_ix in range(new_kv_state.size(0)):
                    new_kv_state[kv_state_ix][vix] = kv_state[kv_state_ix][v['q']]


                beam_seq[t, vix] = v['c']
                beam_seq_logprobs[t, vix] = v['r']
                beam_logprobs_sum[vix] = v['p']
            state = new_state
            visual_key = new_visual_key
            lang_key = new_lang_key
            kv_state = new_kv_state
            return beam_seq, beam_seq_logprobs, beam_logprobs_sum, state, candidates,visual_key,lang_key, kv_state

        opt = kwargs['opt']
        beam_size = opt.get('beam_size', 10)
        group_size = opt.get('group_size', 1)
        diversity_lambda = opt.get('diversity_lambda', 0.5)
        max_ppl = opt.get('max_ppl', 0)
        length_penalty = utils.penalty_builder(opt.get('length_penalty', 'wu_0.5'))

        bdash = beam_size   # beam per group

        # INITIALIZATIONS
        beam_seq_table = torch.LongTensor(self.seq_length, bdash).zero_()
        beam_seq_logprobs_table = torch.FloatTensor(self.seq_length, bdash).zero_()
        beam_logprobs_sum_table = torch.zeros(bdash)

        done_beams_table = []
        state_table = init_state
        logprobs_table = init_logprobs
        # END INIT

        # Chunk elements in the args
        args = list(args)
        # args = [_.chunk(1) if _ is not None else [None] * 1 for _ in args]
        args = [args[i] for i in range(len(args))]
        visual_key,lang_key, kv_state = args[2], args[4], args[5]
        beam_flag = {}
        finish_flag = 0
        for t in range(self.seq_length):

            logprobsf = logprobs_table.data.float()

            logprobsf[:, 1] = logprobsf[:, 1] - 1000


            unaug_logprobsf = add_diversity(beam_seq_table, logprobsf, t, 0, diversity_lambda, bdash)

            beam_seq_table, \
            beam_seq_logprobs_table, \
            beam_logprobs_sum_table, \
            state_table, \
            candidates_divm, \
            visual_key, \
            lang_key, \
            kv_state = beam_step(logprobsf,
                                        unaug_logprobsf,
                                        bdash,
                                        t,
                                        beam_seq_table,
                                        beam_seq_logprobs_table,
                                        beam_logprobs_sum_table,
                                        state_table,
                                        visual_key,
                                        lang_key,
                                        kv_state
                                  )

            for vix in range(bdash):
                if beam_seq_table[t , vix] == 0 or t == self.seq_length - 1:

                    final_beam = {
                        'seq': beam_seq_table[:, vix].clone(),
                        'logps': beam_seq_logprobs_table[:, vix].clone(),
                        'unaug_p': beam_seq_logprobs_table[:, vix].sum().item(),
                        'p': beam_logprobs_sum_table[vix].item()
                    }
                    final_beam['p'] = length_penalty(t+1, final_beam['p'])

                    done_beams_table.append(final_beam)
                    beam_logprobs_sum_table[vix] = -1000

            it = beam_seq_table[t]
            logprobs_table, state_table, args = self.get_logprobs_state(it.cuda(), *(args + [state_table]))

        done_beams_table = [sorted(done_beams_table, key=lambda x: -x['p'])[:bdash]]
        done_beams = reduce(lambda a, b: a + b, done_beams_table)
        visual_key, lang_key, kv_state = args[2], args[4], args[5]
        return done_beams,visual_key,lang_key,kv_state