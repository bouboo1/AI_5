import os
import json
import chardet
import torch
from tqdm import tqdm
from PIL import Image
from sklearn.model_selection import train_test_split


def data_format(input_path, data_dir, output_path):
    data = []
    with open(input_path) as f:
        for line in tqdm(f.readlines(), desc='Formation'):
            guid, label = line.replace('\n', '').split(',')
            text_path = os.path.join(data_dir, (guid + '.txt'))
            if guid == 'guid': continue
            with open(text_path, 'rb') as textf:
                text_byte = textf.read()
                text = text_byte.decode('gb18030')
            text = text.strip('\n').strip('\r').strip(' ').strip()
            data.append({
                'guid': guid,
                'label': label,
                'text': text
            })
    with open(output_path, 'w') as wf:
        json.dump(data, wf, indent=4)


def read_from_file(path, data_dir, only=None):
    data = []
    with open(path) as f:
        json_file = json.load(f)
        for d in tqdm(json_file, desc='Load'):
            guid, label, text = d['guid'], d['label'], d['text']
            if guid == 'guid':
                continue
            if only == 'text':
                img = Image.new(mode='RGB', size=(224, 224), color=(0, 0, 0))
            else:
                img_path = os.path.join(data_dir, (guid + '.jpg'))
                img = Image.open(img_path)
                img.load()

            if only == 'img': text = ''

            data.append((guid, text, img, label))
        f.close()

    return data


# 划分验证集和训练集
def train_val_split(data, val_size=0.2):
    return train_test_split(data, train_size=(1 - val_size), test_size=val_size)


def write_to_file(path, outputs):
    with open(path, 'w') as f:
        for line in tqdm(outputs, desc='Write'):
            f.write(line)
            f.write('\n')
        f.close()


