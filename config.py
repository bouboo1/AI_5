import os

class config:
    root_path = os.getcwd()
    data_dir = os.path.join(root_path, './data/data/')
    train_data_path = os.path.join(root_path, 'data/train.json')
    test_data_path = os.path.join(root_path, 'data/test.json')
    output_path = os.path.join(root_path, 'output')
    output_test_path = os.path.join(output_path, 'test.txt')
    load_model_path = None

    epoch = 1
    learning_rate = 2e-5
    weight_decay = 0
    num_labels = 3
    loss_weight = [1.68, 9.3, 3.36]

    fuse_model_type = 'default'
    only = None
    middle_hidden_size = 64
    attention_nhead = 8
    attention_dropout = 0.4
    fuse_dropout = 0.5
    out_hidden_size = 128

    fixed_text_model_params = False
    bert_name = 'bert-base-multilingual-cased'
    bert_learning_rate = 5e-6
    bert_dropout = 0.2
    fixed_img_model_params = False
    image_size = 224
    fixed_image_model_params = True
    resnet_learning_rate = 5e-6
    resnet_dropout = 0.2
    img_hidden_seq = 64

    checkout_params = {'batch_size': 2, 'shuffle': False}
    train_params = {'batch_size': 2, 'shuffle': True}
    val_params = {'batch_size': 2, 'shuffle': False}
    test_params = {'batch_size': 2, 'shuffle': False}
