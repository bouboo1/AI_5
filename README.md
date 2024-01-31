## 实验五：  多模态情感分析

10215501433 仲韦萱

## Repository structure 仓库结构


```
|-- 5
    |-- requirements.txt
    |-- Trainer.py # 训练器
    |-- data
    |   |-- test.json # 导出文件
    |   |-- test_without_label.txt # 数据源文件
    |   |-- train.json # 导出文件
    |   |-- train.txt # 源文件
    |   |-- data # 原始数据 
    |-- Models
    |   |-- cat.py # 使用cat方法融合的模型
    |   |-- default.py # 使用default方法融合的模型
    |   |-- ImageModel.py # 图像模型
    |   |-- TextModel.py # 文本模型
    |-- output # 存放了训练完毕的模型
    |-- deal.py # 预处理
    |-- config.py # 配置参数
    |-- data_processor.py # 存放了有关实验训练时数据处理的函数
    |-- main.py # 主函数
    |-- test.txt # 预测结果
```

## Setup 开始

运行下面的脚本

```
pip install -r requirements.txt
```

## 运行

训练模型

```shell
python main.py --train --epoch 5 --model bert-base-uncased --type cat
```

若要训练消融模型，请在后面加上--text_only 或 --img_only

预测模型

```shell
python main.py --test --model bert-base-uncased --type cat --load_model_path  + 模型所在位置 （例如output/cat/model.bin）
```
## 参考
https://github.com/RecklessRonan/GloGNN/blob/master/readme.md

注意：数据集过大需要自行下载。