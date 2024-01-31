from tqdm import tqdm
from transformers import AutoTokenizer
from torchvision import transforms
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from sklearn.metrics import  classification_report, accuracy_score


def encoder(data, labelvocab, config):
    labelvocab.add_label('positive')
    labelvocab.add_label('neutral')
    labelvocab.add_label('negative')
    labelvocab.add_label('null')
    tokenizer = AutoTokenizer.from_pretrained(config.bert_name)


    def get_resize(image_size):
        for i in range(20):
            if 2 ** i >= image_size:
                return 2 ** i
        return image_size

    img_transform = transforms.Compose([
        transforms.Resize(get_resize(config.image_size)),
        transforms.CenterCrop(config.image_size),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    guids, encoded_texts, encoded_imgs, encoded_labels = [], [], [], []
    for line in tqdm(data, desc='----- [Encoding]'):
        guid, text, img, label = line
        guids.append(guid)
        text.replace('#', '')
        tokens = tokenizer.tokenize('[CLS]' + text + '[SEP]')
        encoded_texts.append(tokenizer.convert_tokens_to_ids(tokens))
        encoded_imgs.append(img_transform(img))
        encoded_labels.append(labelvocab.label_to_id(label))

    return guids, encoded_texts, encoded_imgs, encoded_labels


def decoder(outputs, labelvocab):
    formated_outputs = ['guid,tag']
    for guid, label in tqdm(outputs, desc='----- [Decoding]'):
        formated_outputs.append((str(guid) + ',' + labelvocab.id_to_label(label)))
    return formated_outputs

def metrix(true_labels, pred_labels):
    print(classification_report(true_labels, pred_labels))
    return accuracy_score(true_labels, pred_labels)


class dataset(Dataset):

    def __init__(self, guids, texts, imgs, labels) -> None:
        self.guids = guids
        self.texts = texts
        self.imgs = imgs
        self.labels = labels

    def __len__(self):
        return len(self.guids)

    def __getitem__(self, index):
        return self.guids[index], self.texts[index], \
            self.imgs[index], self.labels[index]

    # collate_fn = None
    def collate_fn(self, batch):
        guids = [b[0] for b in batch]
        texts = [torch.LongTensor(b[1]) for b in batch]
        imgs = torch.FloatTensor([np.array(b[2]).tolist() for b in batch])
        labels = torch.LongTensor([b[3] for b in batch])
        texts_mask = [torch.ones_like(text) for text in texts]

        paded_texts = pad_sequence(texts, batch_first=True, padding_value=0)
        paded_texts_mask = pad_sequence(texts_mask, batch_first=True, padding_value=0).gt(0)

        return guids, paded_texts, paded_texts_mask, imgs, labels


class LabelVocab:
    UNK = 'UNK'

    def __init__(self) -> None:
        self.label2id = {}
        self.id2label = {}

    def __len__(self):
        return len(self.label2id)

    def add_label(self, label):
        if label not in self.label2id:
            self.label2id.update({label: len(self.label2id)})
            self.id2label.update({len(self.id2label): label})

    def label_to_id(self, label):
        return self.label2id.get(label)

    def id_to_label(self, id):
        return self.id2label.get(id)


class Processor:

    def __init__(self, config) -> None:
        self.config = config
        self.labelvocab = LabelVocab()
        pass

    def __call__(self, data, params):
        return self.to_loader(data, params)

    def encode(self, data):
        return encoder(data, self.labelvocab, self.config)

    def decode(self, outputs):
        return decoder(outputs, self.labelvocab)

    def metric(self, inputs, outputs):
        return metrix(inputs, outputs)

    def to_dataset(self, data):
        dataset_inputs = self.encode(data)
        return dataset(*dataset_inputs)

    def to_loader(self, data, params):
        dataset = self.to_dataset(data)
        return DataLoader(dataset=dataset, **params, collate_fn=dataset.collate_fn)
