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

    with open(os.path.join(data_dir, 'train/validation_list.txt')) as val_list:
        val_files = val_list.readlines()
        val_set = set()
        for entry in val_files:
            r = re.match(pattern, entry)
            if r:
                print r.group(3)
                val_set.add(r.group(3))

if __name__ == '__main__':
    load_data()