import argparse

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sen_max', type=int, default=6,
                        help='number of sentence in a paragraph')
    parser.add_argument('--word_max', type=int, default=30,
                        help='number of words in a sentence')
    parser.add_argument('--weight_sen', type=float, default=5.0,
                        help='weight loss of sentence end')
    parser.add_argument('--weight_word', type=float, default=1.0,
                        help='weight loss of word')
    #  load pretrained word vectors from dense captioning
    parser.add_argument('--fine_tune_embed_after', type=int, default=40,
                        help='after how many epoches fine tune embedding')
    parser.add_argument('--word_vector_file', type=str, default=None,#'data/extract_word_vector.json',
                        help='weight loss of word')
    parser.add_argument('--weight_bias_file', type=str, default=None,
                        help='weight bias from the dense captioning')
    # Data input settings
    parser.add_argument('--input_json', type=str, default='data/cocotalk.json',
                    help='path to the json file containing additional info and vocab')
    parser.add_argument('--input_fc_dir', type=str, default='data/cocotalk_fc',
                    help='path to the directory containing the preprocessed fc feats')
    parser.add_argument('--input_att_dir', type=str, default='data/cocotalk_att',
                    help='path to the directory containing the preprocessed att feats')
    parser.add_argument('--input_label_h5', type=str, default='data/cocotalk_label.h5',
#                    help='path to the h5file containing the preprocessed dataset')
                    help='path to the h5file containing the preprocessed label')
    parser.add_argument('--start_from', type=str, default='save/',
                    help="""continue training from saved model at this path. Path must contain files saved by previous training process: 
                        'infos.pkl'         : configuration;
                        'checkpoint'        : paths to model file(s) (created by tf).
                                              Note: this file contains absolute paths, be careful when moving files around;
                        'model.ckpt-*'      : file(s) with model definition (created by tf)
                    """)

    # Model settings
    parser.add_argument('--topic_dim', type=int, default=1024,  # 512
                        help='size of the topic vector,must be same with word_encoding_size')
    parser.add_argument('--word_encoding_size', type=int, default=512,  # 512
                        help='the encoding size of each token in the vocabulary, must be same with topic_dim')

    parser.add_argument('--att_hid_size', type=int, default=512,  # 512
                        help='size of the attention hidden size')
    parser.add_argument('--feats_pool_dim', type=int, default=1024,  # 512
                        help='size of the topic vector')
    parser.add_argument('--sen_rnn_size', type=int, default=512, #512
                    help='size of the rnn in number of hidden nodes in each layer')
    parser.add_argument('--word_rnn_size', type=int, default=512,  # 512
                        help='size of the rnn in number of hidden nodes in each layer')
    # language attention setting
    parser.add_argument('--caption_rnn_size', type=int, default=512,  # 512
                        help='size of the rnn in number of hidden nodes in dense caption LSTM')
    parser.add_argument('--dense_bidirectional', default=False,
                        help='whether the dense caption encoder is bidirectional')
    parser.add_argument('--per_box', type=int, default=50,  # 512
                        help='the num of regions  per image')

    parser.add_argument('--feats_dim', type=int, default=4096,
                    help='not change')
    parser.add_argument('--stop_value', type=float, default=0.5,
                        help='value of stop')
    # adaptive attention
    parser.add_argument('--max_att_step', type=int, default=4,
                        help='number of epochs')
    parser.add_argument('--epsilon', type=float, default=1e-4,
                        help='number of epochs')
    parser.add_argument('--aat_lambda', type=float, default=1e-4,
                        help='penalty factor on attention steps')


    # Optimization: General
    parser.add_argument('--max_epochs', type=int, default=10,
                    help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                    help='minibatch size')
    parser.add_argument('--grad_clip', type=float, default=5.0, #5.,
                    help='clip gradients at this value')
    parser.add_argument('--feat_drop', type=float, default=0.2,
                        help='strength of dropout in the feats embed')
    parser.add_argument('--lang_drop', type=float, default=0.5,
                    help='strength of dropout in the Language Model RNN')
    parser.add_argument('--beam_size', type=int, default=1,
                    help='used when sample_max = 1, indicates number of beams in beam search. Usually 2 or 3 works well. More is not better. Set this to 1 for faster runtime but a bit worse performance.')

    #Optimization: for the Language Model
    parser.add_argument('--optim', type=str, default='adam',
                    help='what update to use? rmsprop|sgd|sgdmom|adagrad|adam')
    parser.add_argument('--learning_rate', type=float, default=5e-5,#4e-4,
                    help='learning rate')
    parser.add_argument('--learning_rate_decay_start', type=int, default=-1,#-1, 
                    help='at what iteration to start decaying learning rate? (-1 = dont) (in epoch)')
    parser.add_argument('--learning_rate_decay_every', type=int, default=10,
                    help='every how many iterations thereafter to drop LR?(in epoch)')
    parser.add_argument('--learning_rate_decay_rate', type=float, default=0.9,
                    help='every how many iterations thereafter to drop LR?(in epoch)')
#    parser.add_argument('--optim_alpha', type=float, default=0.9,
#                    help='alpha for adam')
    parser.add_argument('--optim_alpha', type=float, default=0.8,
                    help='alpha for adam')
    parser.add_argument('--optim_beta', type=float, default=0.999,
                    help='beta used for adam')
    parser.add_argument('--optim_epsilon', type=float, default=1e-8,
                    help='epsilon that goes into denominator for smoothing')

    
    parser.add_argument('--scheduled_sampling_start', type=int, default=-1, #-1 
                    help='at what iteration to start decay gt probability')
    parser.add_argument('--scheduled_sampling_increase_every', type=int, default=6,
                    help='every how many iterations thereafter to gt probability')
    parser.add_argument('--scheduled_sampling_increase_prob', type=float, default=0.05, 
                    help='How much to update the prob')
    parser.add_argument('--scheduled_sampling_max_prob', type=float, default=0.25, 
                    help='Maximum scheduled sampling prob.')


    # Evaluation/Checkpointing
    parser.add_argument('--val_images_use', type=int, default=5000,#3200
                    help='how many images to use when periodically evaluating the validation loss? (-1 = all)')
    parser.add_argument('--save_checkpoint_every', type=int, default=2500,
                    help='how often to save a model checkpoint (in iterations)?')
    parser.add_argument('--checkpoint_path', type=str, default='save',
                    help='directory to store checkpointed models')
    parser.add_argument('--language_eval', type=int, default=1,
                    help='Evaluate language as well (1 = yes, 0 = no)? BLEU/CIDEr/METEOR/ROUGE_L? requires coco-caption code from Github.')
    parser.add_argument('--losses_log_every', type=int, default=25,
                    help='How often do we snapshot losses, for inclusion in the progress dump? (0 = disable)')       
    parser.add_argument('--load_best_score', type=int, default=1,
                    help='Do we load previous best score when resuming training.')       

    # misc
    parser.add_argument('--id', type=str, default='topdown_rl',
                    help='an id identifying this run/job. used in cross-val and appended when writing progress files')
    parser.add_argument('--train_only', type=int, default=0,
                    help='if true then use 80k, else use 110k')

    args = parser.parse_args()

    # Check if args are valid
    assert args.word_encoding_size > 0, "input_encoding_size should be greater than 0"
    assert args.batch_size > 0, "batch_size should be greater than 0"
    assert args.beam_size > 0, "beam_size should be greater than 0"
    assert args.save_checkpoint_every > 0, "save_checkpoint_every should be greater than 0"
    assert args.losses_log_every > 0, "losses_log_every should be greater than 0"
    assert args.language_eval == 0 or args.language_eval == 1, "language_eval should be 0 or 1"
    assert args.load_best_score == 0 or args.load_best_score == 1, "language_eval should be 0 or 1"
    assert args.train_only == 0 or args.train_only == 1, "language_eval should be 0 or 1"

    return args
