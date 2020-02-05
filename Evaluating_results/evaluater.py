
import os
import tables
import argparse
import torch
import sys

sys.path.append('/Users/sebastiaanscholten/Documents/speech2image-master/PyTorch/functions')

from trainer import flickr_trainer
from encoders import img_encoder, audio_rnn_encoder
from minibatchers import iterate_audio

class personaltrainer(flickr_trainer):
    def audio_batcher(self, data, batch_size, shuffle):
        return iterate_audio(data, batch_size, self.vis, self.cap, shuffle)


parser = argparse.ArgumentParser(description='Create and run an articulatory feature classification DNN')

# args concerning file location
parser.add_argument('-data_loc', type = str, default = '/Users/sebastiaanscholten/Documents/speech2image-master/experiments/Generating_Flickrwords_mfcc/mfcc/words_mfcc_features.h5')

parser.add_argument('-split_loc', type=str,
                    default='/Users/sebastiaanscholten/Documents/speech2image-master/preprocessing/testfolder2/test/dataset.json',
                    help='location of the json file containing the data split information')
parser.add_argument('-results_loc', type=str,
                    default='/Users/sebastiaanscholten/Documents/speech2image-master/PyTorch/flickr_audio/results/',
                    help='location of the json file containing the data split information')
# args concerning training settings
parser.add_argument('-batch_size', type=int, default=10, help='batch size, default: 100')
parser.add_argument('-cuda', type=bool, default=False, help='use cuda, default: True')
# args concerning the database and which features to load
parser.add_argument('-visual', type=str, default='resnet',
                    help='name of the node containing the visual features, default: resnet')
parser.add_argument('-cap', type=str, default='mfcc',
                    help='name of the node containing the audio features, default: mfcc')
parser.add_argument('-gradient_clipping', type=bool, default=True, help='use gradient clipping, default: True')

args = parser.parse_args()

# create config dictionaries with all the parameters for your encoders

audio_config = {'conv': {'in_channels': 39, 'out_channels': 64, 'kernel_size': 6, 'stride': 2,
                         'padding': 0, 'bias': False}, 'rnn': {'input_size': 64, 'hidden_size': 1024,
                                                               'num_layers': 4, 'batch_first': True,
                                                               'bidirectional': True, 'dropout': 0},
                'att': {'in_size': 2048, 'hidden_size': 128, 'heads': 1}}
# automatically adapt the image encoder output size to the size of the caption encoder
out_size = audio_config['rnn']['hidden_size'] * 2 ** audio_config['rnn']['bidirectional'] * audio_config['att']['heads']
image_config = {'linear': {'in_size': 2048, 'out_size': out_size}, 'norm': True}

# open the data file
data_file = tables.open_file(args.data_loc, mode='r+')

# check if cuda is availlable and user wants to run on gpu
cuda = args.cuda and torch.cuda.is_available()
if cuda:
    print('using gpu')
else:
    print('using cpu')


# flickr doesnt need to be split at the root node
def iterate_data(h5_file):
    for x in h5_file.root:
        yield x


f_nodes = [node for node in iterate_data(data_file)]

# split the database into train test and validation sets. default settings uses the json file
# with the karpathy split
test = f_nodes[1:100]
#####################################################

# network modules
img_net = img_encoder(image_config)
cap_net = audio_rnn_encoder(audio_config)

# list all the trained model parameters
models = os.listdir(args.results_loc)
caption_models = [x for x in models if 'caption' in x]
img_models = [x for x in models if 'image' in x]

# run the image and caption retrieval
img_models.sort()
caption_models.sort()

# create a trainer with just the evaluator for the purpose of testing a pretrained model
trainer = personaltrainer(img_net, cap_net, args.visual, args.cap)
trainer.set_audio_batcher()
# optionally use cuda

if cuda:
    trainer.set_cuda()
trainer.set_evaluator([1, 5, 10])

for img, cap in zip(img_models, caption_models):
    epoch = img.split('.')[1]
    # load the pretrained embedders
    trainer.load_cap_embedder(args.results_loc + cap)
    trainer.load_img_embedder(args.results_loc + img)

    # calculate the recall@n
    trainer.set_epoch(epoch)
    trainer.recall_at_n(test, args.batch_size, prepend='test')

