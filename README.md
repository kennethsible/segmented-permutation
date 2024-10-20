# An Artiﬁcial Subword Translation Task
**Ken Sible | [NLP Group](https://nlp.nd.edu)** | **University of Notre Dame**

![model_8.png](experiments/model_8.png)
![model_16.png](experiments/model_16.png)
![model_32.png](experiments/model_32.png)

To generate training/validation data, see [translation.ipynb](/translation.ipynb).

## Training Example
```
$ python translation/main.py --train-data data_bits/train_8_1.tsv --val-data data_bits/val_8_1.tsv --sw-vocab data_bits/vocab_8_1.tsv --model data_bits/model_8_1.pt --log data_bits/model_8_1.log --k 1
```

## Training Reference
```
usage: main.py [-h] --train-data FILE_PATH --val-data FILE_PATH --sw-vocab FILE_PATH --model FILE_PATH --log FILE_PATH [--seed SEED]

options:
  -h, --help            show this help message and exit
  --train-data FILE_PATH
                        parallel training data
  --val-data FILE_PATH  parallel validation data
  --sw-vocab FILE_PATH  subword vocab
  --model FILE_PATH     translation model
  --log FILE_PATH       logger output
  --seed SEED           random seed
```

