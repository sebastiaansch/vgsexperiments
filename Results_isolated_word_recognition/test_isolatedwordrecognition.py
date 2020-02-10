
import os
import tables
import argparse
import torch
import sys
import json

sys.path.append('/Users/sebastiaanscholten/Documents/speech2image-master/PyTorch/functions')
sys.path.append('../')
from helper import create_noun_set
import pandas as pd
import numpy as np

from trainer import flickr_trainer
from evaluate import evaluate
from encoders import img_encoder, audio_rnn_encoder
from minibatchers import iterate_audio
from data_split import split_data_flickr



class evaluating(evaluate):
    def results_at_1(self):
        embeddings_1 = self.caption_embeddings
        embeddings_2 = self.image_embeddings
        results = []
        for index, emb in enumerate(embeddings_1):
            sim = self.dist(emb, embeddings_2)
            idx = torch.argmax(sim)
            results.append(idx)
        return results
    def results_at_n(self,n):
        embeddings_1 = self.caption_embeddings
        embeddings_2 = self.image_embeddings
        results = []
        for index, emb in enumerate(embeddings_1):
            sim = self.dist(emb, embeddings_2)
            sorted, indices = sim.sort(descending = True)
            results.append(indices[0:n])
        return results





    def sep_embed_data(self, speechiterator,imageiterator):
        # set to evaluation mode
        self.embed_function_1.eval()
        self.embed_function_2.eval()

        for speech in speechiterator:
            cap, lengths = speech
            sort = np.argsort(- np.array(lengths))
            cap = cap[sort]
            lens = np.array(lengths)[sort]
            cap = self.dtype(cap)
            cap = self.embed_function_2(cap, lens)
            cap = cap[torch.LongTensor(np.argsort(sort))] #used to be: cap = cap[torch.cuda.LongTensor(np.argsort(sort))]
            try:
                caption = torch.cat((caption, cap.data))
            except:
                caption = cap.data
        for images in imageiterator:
            img = images
            img = img[sort]
            img = self.dtype(img)
            img = self.embed_function_1(img)
            img = img[torch.LongTensor(np.argsort(sort))]
            try:
                image = torch.cat((image, img.data))
            except:
                image = img.data
        self.image_embeddings = image
        self.caption_embeddings = caption

class personaltrainer(flickr_trainer):
    def only_audio_batcher(self, data, batch_size, shuffle):
        return only_iterate_audio(data,batch_size,self.cap,shuffle)
    def only_image_batcher(self, data, batch_size, shuffle):
        return only_iterate_images(data,batch_size,self.vis,shuffle)

    def set_evaluator(self, n):
        self.evaluator = evaluating(self.dtype, self.img_embedder, self.cap_embedder)
        self.evaluator.set_n(n)
    def set_only_audio_batcher(self):
        self.audiobatcher = self.only_audio_batcher
    def set_only_image_batcher(self):
        self.imagebatcher = self.only_image_batcher

    def retrieve_best_image(self,speechdata,imgdata,batch_size):
        speechiterator = self.audiobatcher(speechdata, 5,shuffle=False)
        imageiterator = self.imagebatcher(imgdata, 5, shuffle=False)

        self.evaluator.sep_embed_data(speechiterator,imageiterator)
        return self.evaluator.results_at_1()
    def word_precision_at_n(self, speechdata, imgdata, n, batch_size):
        speechiterator = self.audiobatcher(speechdata, 5,shuffle=False)
        imageiterator = self.imagebatcher(imgdata, 5, shuffle=False)

        self.evaluator.sep_embed_data(speechiterator,imageiterator)
        results = self.evaluator.results_at_n(n)
        return check_word_occurence(speechdata,imgdata,results)

def only_iterate_audio(f_nodes, batchsize, audio, shuffle=True):
    frames = 2048
    if shuffle:
        # optionally shuffle the input
        np.random.shuffle(f_nodes)
    for start_idx in range(0, len(f_nodes) - batchsize + 1, batchsize):
        # take a batch of nodes of the given size
        excerpt = f_nodes[start_idx:start_idx + batchsize]
        speech = []
        lengths = []
        for ex in excerpt:
            # extract and append the visual features
            # retrieve the audio features
            sp = eval('ex.' + audio + '._f_list_nodes()[0].read().transpose()')
            # padd to the given output size
            n_frames = sp.shape[1]
            if n_frames < frames:
                sp = np.pad(sp, [(0, 0), (0, frames - n_frames)], 'constant')
            # truncate to the given input size
            if n_frames > frames:
                sp = sp[:, :frames]
                n_frames = frames
            lengths.append(n_frames)
            speech.append(sp)

        max_length = max(lengths)
        # reshape the features and recast as float64
        speech = np.float64(speech)
        # truncate all padding to the length of the longest utterance
        speech = speech[:, :, :max_length]
        # reshape the features into appropriate shape and recast as float32
        speech = np.float64(speech)
        yield speech, lengths

def only_iterate_images(f_nodes, batchsize, visual, shuffle=True):
    if shuffle:
        # optionally shuffle the input
        np.random.shuffle(f_nodes)
    for start_idx in range(0, len(f_nodes) - batchsize + 1, batchsize):
        # take a batch of nodes of the given size
        excerpt = f_nodes[start_idx:start_idx + batchsize]
        images = []
        for ex in excerpt:
            # extract and append the visual features
            images.append(eval('ex.' + visual + '._f_list_nodes()[0].read()'))
            # retrieve the audio features
            # padd to the given output size
        # reshape the features and recast as float64
        images = np.float64(images)
        yield images

def check_word_occurence(testedwords,testset,results):
    path = "/Users/sebastiaanscholten/Documents/speech2image-master/experiments/data/imagecaptiontextdictionary.json"

    with open(path, "r") as json_file:
        data = json.load(json_file)
    returnedcorrectly = []
    for idx,result in enumerate(results):
        testword = testedwords[idx]._v_name.replace('flickr_', '')
        resultslist = []
        for res in result:
            filename = testset[res]._v_name.replace('flickr_', '')+".jpg"
            for word in data[filename]:
                found = False
                if word == testword:
                    resultslist.append(1)
                    found = True
                    break
            if found == False:
                resultslist.append(0)
        returnedcorrectly.append(resultslist)
    return returnedcorrectly, results

parser = argparse.ArgumentParser(description='Create and run an articulatory feature classification DNN')

# args concerning file location
parser.add_argument('-data_loc', type = str, default = '/Users/sebastiaanscholten/Documents/speech2image-master/experiments/Generating_Flickrwords_mfcc/mfcc/words_mfcc_features.h5')

parser.add_argument('-flickr_loc', type = str, default = '/Users/sebastiaanscholten/Documents/speech2image-master/preprocessing/prep_data/flickr_features_27jan_fixed.h5',
                    help = 'location of the Flickr feature file, default: /prep_data/flickr_features.h5')

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
flickr_file = tables.open_file(args.flickr_loc, mode="r+")
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


nouns_txt = pd.read_csv("/Users/sebastiaanscholten/Documents/speech2image-master/experiments/data/testwords.txt", header=None)
nouns = nouns_txt.iloc[:,0].values.tolist()

f_nodes_mfcc = create_noun_set(nouns,data_file)

f_nodes_flickr = [node for node in iterate_data(flickr_file)]

# split the database into train test and validation sets. default settings uses the json file
# with the karpathy split
train, test, val = split_data_flickr(f_nodes_flickr, args.split_loc)


mfcc_test = f_nodes_mfcc
images_test = test


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
trainer.set_only_audio_batcher()
trainer.set_only_image_batcher()
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

    resultsat10, whichimages = trainer.word_precision_at_n(mfcc_test, images_test, 10, args.batch_size)
    for idx,result in enumerate(resultsat10):
        print("For word: ", mfcc_test[idx]._v_name.replace('flickr_', ''), "we retrieved: ", result, "\n")
        for res in whichimages[idx]:
            print(images_test[res]._v_name.replace('flickr_', '')+".jpg")



    print("done!")
    print("okay")
