import tensorflow as tf
from glob import glob
import re
import os
data_dir = './data'
POSSIBLE_LABELS = 'yes no up down left right on off stop go silence unknown'.split()
id2name = {i: name for i, name in enumerate(POSSIBLE_LABELS)}
name2id = {name: i for i, name in enumerate(POSSIBLE_LABELS)}

def load_data():
    pattern = re.compile("(.+\/)?(\w+)\/([^_]+)_.+wav")
    all_files = glob(os.path.join(data_dir, 'train/audio/*/*wav'))
    print all_files

if __name__ == '__main__':
    print tf.__version__