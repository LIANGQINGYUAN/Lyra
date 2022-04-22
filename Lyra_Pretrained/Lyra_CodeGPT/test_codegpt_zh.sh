export CUDA_VISIBLE_DEVICES=7
LANG=python
DATADIR=./data
OUTPUTDIR=./save_codegpt_zh
PRETRAINDIR=./save_codegpt_zh/checkpoint-5000-0.31
LOGFILE=./log/zh_codegpt_text2code_lyra_infer.log

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