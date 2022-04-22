export CUDA_VISIBLE_DEVICES=4
output_dir=./save_gcb
log_file=./log/log_test_graphcodebert.txt

python run.py \
    	--do_test \
	--model_type roberta \
	--model_name_or_path roberta-base \
	--config_name roberta-base \
	--tokenizer_name roberta-base  \
	--load_model_path $output_dir/checkpoint-best-bleu/pytorch_model.bin \
	--dev_filename ../data/valid.lyra.csv \
	--test_filename ../data/test.lyra.csv \
	--log_file $log_file\
	--output_dir $output_dir \
	--max_source_length 512 \
	--max_target_length 512 \
	--beam_size 10 \
	--eval_batch_size 8 