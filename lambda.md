Installation
===

```bash
sudo pip3 install virtualenv

sudo apt-get install python3-dev

cd bert
virtualenv -p /usr/bin/python3.6 venv-bert
. venv-bert/bin/activate


```

Data
===

```bash
# Pretrain
wget http://mattmahoney.net/dc/enwik8.zip

# unzip to bert path
unzip enwik8.zip
rm enwik8.zip

export BERT_BASE_DIR=./uncased_L-12_H-768_A-12

CUDA_VISIBLE_DEVICES=0 python create_pretraining_data.py \
  --input_file=./enwik8 \
  --output_file=/tmp/tf_examples.tfrecord \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --do_lower_case=True \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --masked_lm_prob=0.15 \
  --random_seed=12345 \
  --dupe_factor=5
```

```bash
# Question and answer
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json
```



Train
===

```bash
# Pretrain
CUDA_VISIBLE_DEVICES=0 python run_pretraining.py \
  --input_file=./tf_examples.tfrecord \
  --output_dir=./pretraining_output \
  --do_train=True \
  --do_eval=True \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --train_batch_size=32 \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --num_train_steps=200000 \
  --num_warmup_steps=10 \
  --learning_rate=2e-5
```


```bash

export SQUAD_DIR=./squad

# Question and answer
CUDA_VISIBLE_DEVICES=0 python run_squad.py \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --do_train=True \
  --train_file=$SQUAD_DIR/train-v1.1.json \
  --do_predict=False \
  --train_batch_size=1 \
  --learning_rate=3e-5 \
  --num_train_epochs=100.0 \
  --max_seq_length=384 \
  --doc_stride=128 \
  --output_dir=/tmp/squad_base/
```

__Pretrain__

Memory Requirement

| Batch Size | Memory  |
|---|---|
| train_batch_size=8| 6GB |
| train_batch_size=16 | 8GB |
| train_batch_size=32 | 11GB |
| train_batch_size=64 | 24GB |
| train_batch_size=128 | 32GB |

Throughput (examples/sec) 

|   | 2060  | 2070  | 2080  |  1080 Ti | 2080 Ti | TitanRTX | Quadro RTX 6000 | V100 | Quadro RTX 8000 |
|---|---|---|---|---|---|---|---|---|---|
| train_batch_size=8 | 33.52 | 40.32 | 45.75 | 43.62 | 59.62 | 66.35 | 64.91 |  | 61.15  |
| train_batch_size=16 | OOM | 47.26 | 57.73 | 52.93 | 71.18 | 80.39 | 77.43 |  | 73.66 |
| train_batch_size=32 | OOM | OOM | OOM | 59.64 | 82.71 | 94.42 | 92.30 |  | 89.45 |
| train_batch_size=64 | OOM | OOM | OOM | OOM | OOM | 102.38 | 98.28 |  | 90.81 |
| train_batch_size=128 | OOM | OOM | OOM | OOM | OOM | OOM | OOM | 109.34 | 93.57 |


__Question and Answer__

Memory Requirement

| Batch Size | Memory  |
|---|---|
| train_batch_size=1 | 6GB |
| train_batch_size=6 | 8GB |
| train_batch_size=12 | 24GB  |
| train_batch_size=24 | 24GB |
| train_batch_size=48 | 48GB |

Throughput (examples/sec) 

|   | 2060  | 2070  | 2080  |  1080 Ti | 2080 Ti | TitanRTX | Quadro RTX 6000 | V100 | Quadro RTX 8000 |
|---|---|---|---|---|---|---|---|---|---|
| train_batch_size=1 | 7.13 | 8.68 | 9.51 | 8.66 | 12.1127 | 13.48 | 13.34 | | 12.16 | 
| train_batch_size=6 | OOM | 15.21 | 18.07 | 16.88 | 22.42 | 25.89 | 25.23 | | 23.73 | 
| train_batch_size=12 | OOM | OOM | OOM | OOM | OOM | 29.2432 | 27.42 | | 24.94 | 
| train_batch_size=24 | OOM | OOM | OOM | OOM | OOM | 30.46 | 29.35 | 32.393 | 26.28 | 
| train_batch_size=48 | OOM | OOM | OOM | OOM | OOM | OOM | OOM | OOM | 26.78 | 


Notes:
Umcomment Line 1194 - 1213 to generate tfrecords
