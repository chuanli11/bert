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

Train
===

```bash
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


Memory Requirement

| Batch Size | Memory  |
|---|---|
| train_batch_size=8| 6GB |
| train_batch_size=16 | 8GB |
| train_batch_size=32 | |
| train_batch_size=64 |  |
| train_batch_size=128 |  |

Throughput (examples/sec) 

|   | 2060  | 2070  | 2080  |  1080 Ti | 2080 Ti | TitanRTX | Quadro RTX 6000 | V100 | Quadro RTX 8000 |
|---|---|---|---|---|---|---|---|---|---|
| train_batch_size=8 | 33.52 | 40.32 |  |  | 59.6221 | 66.35 |  |  |  |
| train_batch_size=16 | OOM | 47.2642 |  |  | 71.18 | 80.39 |  |  |  |
| train_batch_size=32 | OOM | OOM | OOM |  | 82.71 | 94.42 |  |  |  |
| train_batch_size=64 | OOM | OOM | OOM | OOM | OOM | 102.38 |  |  |  |
| train_batch_size=128 | OOM | OOM | OOM | OOM | OOM | OOM |  |  |  |
