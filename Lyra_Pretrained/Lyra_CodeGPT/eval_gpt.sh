export CUDA_VISIBLE_DEVICES=0
LANG=python
DATADIR=./data
OUTPUTDIR=./save_gpt
PRETRAINDIR=./save_gpt/checkpoint-13000-0.245
LOGFILE=./log/gpt2_text2code_lyra_eval.log

python -u run.py \
        --data_dir=$DATADIR \
        --langs=$LANG \
        --output_dir=$OUTPUTDIR \
        --pretrain_dir=$PRETRAINDIR \
        --log_file=$LOGFILE \
        --model_type=gpt2 \
        --block_size=512 \
        --do_eval \
        --logging_steps=100 \
        --seed=42