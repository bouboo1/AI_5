import os
import torch
import argparse
from config import config
from deal import data_format, read_from_file, train_val_split, write_to_file
from data_processor import Processor
from Trainer import Trainer
import matplotlib.pyplot as plt

torch.cuda.empty_cache()
# Train
def train(processor, trainer):
    data_format(os.path.join(config.root_path, './data/train.txt'),
                os.path.join(config.root_path, './data/data'), os.path.join(config.root_path, './data/train.json'))
    data = read_from_file(config.train_data_path, config.data_dir, config.only)
    train_data, val_data = train_val_split(data)
    train_loader = processor(train_data, config.train_params)
    val_loader = processor(val_data, config.val_params)

    train_acc = []
    valid_acc = []

    best_acc = 0.0
    epoch = config.epoch
    for e in range(epoch):
        print('-' * 20 + ' ' + 'Epoch ' + str(e + 1) + ' ' + '-' * 20)
        tloss, tloss_list, train_accu = trainer.train(train_loader)
        print('Train Loss: {}'.format(tloss))
        vloss, valid_accu = trainer.valid(val_loader)
        print('Valid Loss: {}'.format(vloss))
        print('Train Acc: {}'.format(train_accu))
        print('Valid Acc: {}'.format(valid_accu))
        train_acc.append(train_accu)
        valid_acc.append(valid_accu)
        if valid_accu > best_acc:
            best_acc = valid_accu
            save_model(config.output_path, config.fuse_model_type, model)
            print('Model Saved')
        print()

    plt.plot(train_acc, label="train_accuracy")
    plt.plot(valid_acc, label="valid_accuracy")
    plt.title(model.__class__.__name__)
    plt.xlabel("Epoch")
    plt.xticks(range(epoch), range(1, epoch + 1))
    plt.ylabel("Accuracy")
    plt.ylim(ymin=0, ymax=1)
    plt.legend()
    plt.show()
    print("Best Valid_acc:",best_acc)
    print("learning rate",config.learning_rate)
    

def test(processor, trainer):
    data_format(os.path.join(config.root_path, './data/test_without_label.txt'),
                os.path.join(config.root_path, './data/data'), os.path.join(config.root_path, './data/test.json'))
    test_data = read_from_file(config.test_data_path, config.data_dir, config.only)
    test_loader = processor(test_data, config.test_params)

    if config.load_model_path is not None:
        model.load_state_dict(torch.load(config.load_model_path))

    outputs = trainer.predict(test_loader)
    formated_outputs = processor.decode(outputs)
    write_to_file(config.output_test_path, formated_outputs)


def save_model(output_path, model_type, model):
    output_model_dir = os.path.join(output_path, model_type)
    if not os.path.exists(output_model_dir):
        os.makedirs(output_model_dir)
    model_to_save = model.module if hasattr(model, 'module') else model
    output_model_file = os.path.join(output_model_dir, "model.bin")
    torch.save(model_to_save.state_dict(), output_model_file)

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Args
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--model', default='bert-base-multilingual-cased', type=str)
    parser.add_argument('--type', default='cat', type=str)
    parser.add_argument('--lr', default=5e-5, type=float)
    parser.add_argument('--weight_decay', default=1e-2, type=float)
    parser.add_argument('--epoch', default=10, type=int)

    parser.add_argument('--test', action='store_true')
    parser.add_argument('--load_model_path', default=None, type=str)
    parser.add_argument('--text_only', action='store_true')
    parser.add_argument('--img_only', action='store_true')

    args = parser.parse_args()
    config.learning_rate = args.lr
    config.weight_decay = args.weight_decay
    config.epoch = args.epoch
    config.bert_name = args.model
    config.type = args.type
    config.load_model_path = args.load_model_path
    config.only = 'img' if args.img_only else None
    config.only = 'text' if args.text_only else None
    if args.img_only and args.text_only:
        config.only = None

    # Initialization
    processor = Processor(config)

    if config.type == 'cat':
        from models.cat import Model
    elif config.type == 'default':
        from models.default import Model

    model = Model(config)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # device = torch.device("cpu")

    trainer = Trainer(config, processor, model, device)

    if args.train:
        train(processor, trainer)

    if args.test:
        if args.load_model_path is None and not args.do_train:
            print('Model Path Error')
        else:
            test(processor, trainer)
