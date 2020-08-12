# Train

To train, use `train.py`

```
usage: train.py [-h] [-lr LR] [-prob PROB] [-bs BS] [-fold FOLD] [-gpu GPU]
                [-iter_time ITER_TIME] [-data DATA] [-attn]

optional arguments:
  -h, --help            show this help message and exit
  -lr LR                learning rate
  -prob PROB            dropout rate of history mask
  -bs BS                batch size in training
  -fold FOLD            fold
  -gpu GPU              gpu id to use
  -iter_time ITER_TIME  number of iterations for the IOM
  -data DATA            path to the dataset folder
  -attn                 whether or not to separate
```

# Evaluate

To evaluate a trained model, use `val.py`

```
usage: val.py [-h] [-fold FOLD] [-gpu GPU] [-iter_time ITER_TIME] [-w W]
              [-d D] [-s S] [-a] [-p P]

optional arguments:
  -h, --help            show this help message and exit
  -fold FOLD            fold
  -gpu GPU              gpu id to use
  -iter_time ITER_TIME  number of iterations in IOM
  -w W                  path to weight file
  -d D                  path to dataset
  -s S                  random seed
  -a                    use attention or not
  -p P                  number of exps
```

# Visualize

To make inference on one sample of the PASCAL-5i and visualize, use `visualize.py`

```
usage: visualize.py [-h] [--gpus GPUS] [--weight WEIGHT] [--root ROOT]
                    [--refid REFID] [--queid QUEID] [--classid CLASSID]
                    [--niters NITERS] [--a]

optional arguments:
  -h, --help         show this help message and exit
  --gpus GPUS        gpu(s) to be used
  --weight WEIGHT    path to pretrained weights
  --root ROOT        root folder of the PASCAL-5i
  --refid REFID      id of reference image
  --queid QUEID      id of the query image
  --classid CLASSID  id of the semantic class
  --niters NITERS    number of iterations for IOM
  --a                separate attention or not
```