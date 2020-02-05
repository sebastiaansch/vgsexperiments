import tables
import argparse

parser = argparse.ArgumentParser(description='Create and run an articulatory feature classification DNN')
parser.add_argument('-data_loc', type = str, default = '/Users/sebastiaanscholten/Documents/speech2image-master/experiments/Generating_Flickrwords_mfcc/mfcc/words_mfcc_features.h5')

args = parser.parse_args()

data_file = tables.open_file(args.data_loc, mode='r+')

def iterate_data(h5_file):
    for x in h5_file.root:
        yield x

f_nodes = [node for node in iterate_data(data_file)]

x = f_nodes[0:100]

print("okay")