export CUDA_VISIBLE_DEVICES=5
LANG=python
DATADIR=./data
OUTPUTDIR=./save_adapted_zh
PRETRAINDIR=./save_adapted_zh/checkpoint-9500-0.275
LOGFILE=./log/zh_adapted_text2code_lyra_infer.log

python -u run.py \
        --data_dir=$DATADIR \
        --langs=$LANG \
        --nl_langs=zh \
        --output_dir=$OUTPUTDIR \
        --pretrain_dir=$PRETRAINDIR \
        --log_file=$LOGFILE \
        --model_type=gpt2 \
        --block_size=512 \
        --do_infer \
        --logging_steps=100 \
        --seed=42