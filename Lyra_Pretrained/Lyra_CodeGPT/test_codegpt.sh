export CUDA_VISIBLE_DEVICES=6
LANG=python
DATADIR=./data
OUTPUTDIR=./save_codegpt
PRETRAINDIR=./save_codegpt/checkpoint-3000-0.295
LOGFILE=./log/codegpt_text2code_lyra_infer.log

python -u run.py \
        --data_dir=$DATADIR \
        --langs=$LANG \
        --output_dir=$OUTPUTDIR \
        --pretrain_dir=$PRETRAINDIR \
        --log_file=$LOGFILE \
        --model_type=gpt2 \
        --block_size=512 \
        --do_infer \
        --logging_steps=100 \
        --seed=42