import torch
import numpy as np
from torch.autograd import Variable
import torch.optim as optim
import logging
import time
import os
from six.moves import cPickle
import misc.utils as utils
import eval_utils
import opts
# from tensorboardX import SummaryWriter
from dataloader import DataLoader
from models.HRNN import HRNN
from misc.rewards import get_self_critical_reward

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
logging.basicConfig(level=logging.INFO,
                    filename='loss_score.log',
                    filemode='w',
                    format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s')
def update_lstm(layer,flag):
    layer.weight_ih.requires_grad =flag
    layer.weight_hh.requires_grad =flag
    layer.bias_ih.requires_grad = flag
    layer.bias_hh.requires_grad = flag

def update_linear(layer, flag):
    layer.weight.requires_grad = flag
    layer.bias.requires_grad = flag
def train(opt):
    loader = DataLoader(opt)
    # writer = SummaryWriter('log')
    opt.vocab_size = loader.vocab_size
    opt.ix_to_word = loader.get_vocab()
    infos = {}
    if opt.start_from is not None:
        with open(os.path.join(opt.start_from,'infos.pkl'),'rb') as f:
            infos = cPickle.load(f)
            save_model_opt = infos['opt']
            need_be_same = ['word_rnn_size', 'sen_rnn_size', 'topic_dim']
            for checkme in need_be_same:
                assert vars(save_model_opt)[checkme] == vars(opt)[
                    checkme], "Command line argument and saved model disagree on '%s' " % checkme
    iteration = infos.get('iter', 0)
    epoch = infos.get('epoch', 0)


    loader.iterators = infos.get('iterators', loader.iterators)
    loader.split_ix = infos.get('split_ix', loader.split_ix)

    if opt.load_best_score == 1:
        best_val_score = infos.get('best_val_score', None)
    model = HRNN(opt)
    model.cuda()
    if vars(opt).get('start_from', None) is not None:
        # check if all necessary files exist
        assert os.path.isdir(opt.start_from), " %s must be a a path" % opt.start_from
        assert os.path.isfile(os.path.join(opt.start_from,
                                           "infos.pkl")), "infos.pkl file does not exist in path %s" % opt.start_from
        model.load_state_dict(torch.load(os.path.join(opt.start_from, 'model.pth')))
    update_lr_flag = True
    model.train()
    crit = utils.Criterion(opt)
    rl_crit = utils.RewardCriterion()
    optimizer = optim.Adam(model.parameters(),lr=opt.learning_rate)
    if vars(opt).get('start_from',None) is not None and os.path.isfile(os.path.join(opt.start_from,'optimizer.pth')):
        print('load optimizer.pth')
        optimizer.load_state_dict(torch.load(os.path.join(opt.start_from,'optimizer.pth')))
    eval_kwargs = {'split': 'val', 'dataset': opt.input_json, 'verbose': True}
    eval_kwargs.update(vars(opt))
    epoch_start = time.time()
    loss_epoch = []
    while True:
        if update_lr_flag:
            if epoch > opt.learning_rate_decay_start and opt.learning_rate_decay_start >=0:
                frac = (epoch-opt.learning_rate_decay_start) // opt.learning_rate_decay_every
                decay_factor = opt.learning_rate_decay_rate**frac
                opt.current_lr = opt.learning_rate * decay_factor
                utils.set_lr(optimizer,opt.current_lr)
            else:
                opt.current_lr = opt.learning_rate
            if epoch  > opt.scheduled_sampling_start and opt.scheduled_sampling_start >= 0:
                frac = (epoch-opt.scheduled_sampling_start) // opt.scheduled_sampling_increase_every
                opt.ss_prob = min(opt.scheduled_sampling_increase_prob*frac, opt.scheduled_sampling_max_prob)
                model.ss_prob = opt.ss_prob
            if opt.self_critical_after != -1 and epoch >= opt.self_critical_after:
                sc_flag = True
            else:
                sc_flag = False
            # sc_flag = True
            update_lr_flag = False

        start = time.time()
        torch.cuda.synchronize()
        data = loader.get_batch('train')

        feats, labels, masks, sent_place, dense_labels,dense_lengths = utils.input_data(data)

        optimizer.zero_grad()
        if not sc_flag:
            predict_word, predict_end,all_att_cost = model(feats,labels,masks,dense_labels,dense_lengths)
            loss_word, loss_end,loss_att, loss = crit(predict_word, predict_end, labels[:, :, 1:], sent_place, masks[:, :, 1:],
                                             all_att_cost, loader)
        else:

            gen_result, sample_logprobs = model.sample(feats, dense_labels, dense_lengths, opt={'sample_max': 0})
            reward = get_self_critical_reward(model,feats,dense_labels,dense_lengths, data, gen_result)
            loss = rl_crit(sample_logprobs, gen_result, Variable(torch.from_numpy(reward).float().cuda()))

        loss.backward()
        loss_epoch.append(loss.item())
        total_norm = 0

        torch.nn.utils.clip_grad_norm(model.parameters(), opt.grad_clip)
        optimizer.step()
        train_loss = loss.item()
        torch.cuda.synchronize()
        end = time.time()

        if iteration % 50 == 0 and not sc_flag:
            logging.info("iter {}  epoch {} , train_loss {:.3f},loss_word {:.3f},loss_end {:.3f},loss_att {:.3f}, total_norm = {:.3f}, time/batch = {:.3f}" \
                    .format(iteration, epoch, train_loss, loss_word.item(), loss_end.item(),loss_att.item(), total_norm, end - start))
            print(
                "iter {}  epoch {} , train_loss {:.3f},loss_word {:.3f},loss_end {:.3f},loss_att {:.3f}, total_norm = {:.3f}, time/batch = {:.3f}" \
                    .format(iteration, epoch, train_loss, loss_word.item(), loss_end.item(),loss_att.item(), total_norm, end - start))
            # writer.add_scalar('train/train_loss', train_loss, iteration)
        if iteration % 50 == 0 and  sc_flag:  #
            logging.info(
                "iter {}  epoch {} , train_loss {:.3f},avg_reward {:.3f}, total_norm = {:.3f}, time/batch = {:.3f}" \
                .format(iteration, epoch, train_loss, np.mean(reward[:, 0]), total_norm, end - start))
            print(
                "iter {}  epoch {} , train_loss {:.3f},avg_reward {:.3f}, total_norm = {:.3f}, time/batch = {:.3f}" \
                .format(iteration, epoch, train_loss, np.mean(reward[:, 0]), total_norm, end - start))
            # writer.add_scalar('train/train_loss', train_loss, iteration)
            # writer.add_scalar('train/avg_reward', np.mean(reward[:, 0]), iteration)




        iteration = iteration+1
        if iteration % opt.save_checkpoint_every == 0:
           eval_kwargs = {'split':'val', 'dataset':opt.input_json, 'verbose':True}
           eval_kwargs.update(vars(opt))
           val_loss_word, val_loss_end, val_loss, predictions,lang_stats = eval_utils.eval_split(model,crit, loader, eval_kwargs)
           logging.info(
               "Bleu_1 is {0:.3f}, Bleu_2 is {1:.3f},Bleu_3 is {2:.3f}, Bleu_4 is {3:.3f},METEOR is {4:.3f}, ROUGE_L is {5:.3f}, CIDEr is {6:.3f} " \
               .format(lang_stats['Bleu_1'], lang_stats['Bleu_2'], lang_stats['Bleu_3'], lang_stats['Bleu_4'],
                       lang_stats['METEOR'], lang_stats['ROUGE_L'], lang_stats['CIDEr']))
           # writer.add_scalar('val_score/Bleu1', lang_stats['Bleu_1'], iteration)
           # writer.add_scalar('val_score/Bleu4', lang_stats['Bleu_4'], iteration)
           # writer.add_scalar('val_score/METEOR', lang_stats['METEOR'], iteration)
           # writer.add_scalar('val_score/CIDEr', lang_stats['CIDEr'], iteration)

           # writer.add_scalar('val/val_loss', val_loss, iteration)
           # writer.add_scalar('val/loss_word', val_loss_word.item(), iteration)
           # writer.add_scalar('val/loss_end', val_loss_end.item(), iteration)

           if opt.language_eval == 1:
               current_score = lang_stats['Bleu_4']

           best_flag = False
           if True:
               if best_val_score is None or current_score > best_val_score:
                   best_val_score = current_score
                   best_flag = True
               if sc_flag:
                   opt.checkpoint_path = opt.checkpoint_path_scst
               else:
                   opt.checkpoint_path = opt.checkpoint_path_xe
               if not os.path.exists(opt.checkpoint_path):
                   os.mkdir(opt.checkpoint_path)
               checkpoint_path = os.path.join(opt.checkpoint_path, 'model.pth')
               torch.save(model.state_dict(), checkpoint_path)
               print("model saved to {}".format(checkpoint_path))

               optimizer_path = os.path.join(opt.checkpoint_path, 'optimizer.pth')
               torch.save(optimizer.state_dict(), optimizer_path)

               infos['iter'] = iteration
               infos['epoch'] = epoch
               infos['iterators'] = loader.iterators
               infos['split_ix'] = loader.split_ix
               infos['best_val_score'] = best_val_score
               infos['opt'] = opt
               infos['vocab'] = loader.get_vocab()

               with open(os.path.join(opt.checkpoint_path, 'infos.pkl'), 'wb') as f:
                   cPickle.dump(infos, f)
               if best_flag:
                   checkpoint_path = os.path.join(opt.checkpoint_path, 'model-best.pth')
                   torch.save(model.state_dict(), checkpoint_path)
                   print("model saved to {}".format(checkpoint_path))

                   optimizer_path = os.path.join(opt.checkpoint_path, 'optimizer-best.pth')
                   torch.save(optimizer.state_dict(), optimizer_path)
                   with open(os.path.join(opt.checkpoint_path, 'infos-best.pkl'), 'wb') as f:
                       cPickle.dump(infos, f)

        if data['bounds']['wrapped']:
            loss_per = np.mean(np.array(loss_epoch))
            loss_epoch = []

            epoch += 1
            update_lr_flag = True
            print("epoch: " + str(epoch) + " during: " + str(time.time() - epoch_start))
            epoch_start = time.time()

        if epoch >= opt.max_epochs and opt.max_epochs != -1:
            break

def main():
    opt = opts.parse_opt()
    opt.batch_size=50
    opt.learning_rate= 5e-4
    opt.learning_rate_decay_start = 1
    opt.scheduled_sampling_start = 1
    opt.save_checkpoint_every = 200
    opt.val_images_use=5000
    opt.max_epochs = 90
    opt.start_from = None
    opt.self_critical_after = 60
    opt.language_eval = 1
    opt.input_json = 'data/paratalk.json'
    opt.input_label_h5 = 'data/paratalk_label.h5'
    opt.input_att_dir = '/home/sdb1/dataset/image/VG_data/VG_feature.h5'
    opt.checkpoint_path_xe = 'save_xe'
    opt.checkpoint_path_scst = 'save_scst'
    train(opt)
main()

