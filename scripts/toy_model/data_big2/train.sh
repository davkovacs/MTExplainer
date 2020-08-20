#! /bin/bash

dataset=/home/dpk25/rds/hpc-work/toy_model/data_big2/
save_model=/home/dpk25/rds/hpc-work/toy_model/reduction_big2/

python /home/dpk25/MolecularTransformer2/train.py -data ${dataset}  \
	-save_model ${save_model}toy_model \
	-seed 42 -gpu_ranks 0 -save_checkpoint_steps 10000 -keep_checkpoint -1 \
	-train_steps 350000 -param_init 0 -param_init_glorot -max_generator_batches 32 \
	-batch_size 512 -valid_batch_size 8 -batch_type tokens -normalization tokens -max_grad_norm 0 \
	-accum_count 4 -optim adam -adam_beta1 0.9 -adam_beta2 0.998 -decay_method noam \
	-warmup_steps 6000 -learning_rate 2 -label_smoothing 0.0 -report_every 1000 \
	-layers 4 -rnn_size 256 -word_vec_size 256 -encoder_type transformer \
	-decoder_type transformer -dropout 0.1 -position_encoding -share_embeddings \
	-global_attention general -global_attention_function softmax \
	-self_attn_type scaled-dot -heads 8 -transformer_ff 2048 \
	#-train_from ${save_model}toy_model_step_125000.pt \
