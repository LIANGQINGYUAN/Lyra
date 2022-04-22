export CUDA_VISIBLE_DEVICES=4
pretrained_model=microsoft/graphcodebert-base
output_dir=./save_gcb
log_file=./log/log_train_graphcodebert.txt

python run.py \
	--do_train \
	--do_eval \
	--model_type roberta \
	--model_name_or_path $pretrained_model \
	--config_name roberta-base \
	--tokenizer_name roberta-base \
	--train_filename ../data/train.lyra.csv \
	--dev_filename ../data/valid.lyra.csv \
	--output_dir $output_dir \
	--log_file $log_file\
	--max_source_length 512 \
	--max_target_length 512 \
	--beam_size 5 \
	--train_batch_size 8 \
	--eval_batch_size 8 \
	--learning_rate 5e-5 \
	--train_steps 50000 \
	--eval_steps 1000