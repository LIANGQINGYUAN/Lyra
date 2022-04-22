export CUDA_VISIBLE_DEVICES=3
LANG=python
DATADIR=../data
OUTPUTDIR=./save_codegpt
PRETRAINDIR=microsoft/CodeGPT-small-py    # will download pre-trained CodeGPT model # https://huggingface.co/microsoft
LOGFILE=./log/codegpt_text2code_lyra.log
PER_NODE_GPU=1       # modify YOUR_GPU_NUM

python -m torch.distributed.launch --nproc_per_node=$PER_NODE_GPU run.py \
        --data_dir=$DATADIR \
        --langs=$LANG \
        --output_dir=$OUTPUTDIR \
        --pretrain_dir=$PRETRAINDIR \
        --log_file=$LOGFILE \
        --model_type=gpt2 \
        --block_size=512 \
        --do_train \
        --node_index 0 \
        --gpu_per_node $PER_NODE_GPU \
        --learning_rate=5e-5 \
        --weight_decay=0.01 \
        --evaluate_during_training \
        --per_gpu_train_batch_size=6 \
        --per_gpu_eval_batch_size=12 \
        --gradient_accumulation_steps=2 \
        --num_train_epochs=100 \
        --logging_steps=100 \
        --save_steps=500 \
        --overwrite_output_dir \
        --seed=42