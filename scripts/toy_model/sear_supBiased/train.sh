#! /bin/bash

dataset=/home/dpk25/rds/hpc-work/toy_model/data_sear_supBiased/
save_model=/home/dpk25/rds/hpc-work/toy_model/sear_models_supBiased/

python /home/dpk25/MolecularTransformer2/train.py -data ${dataset}  \
	-save_model ${save_model}toy_model \
	-seed 42 -gpu_ranks 0 -save_checkpoint_steps 5000 -keep_checkpoint -1 \
	-train_steps 150000 -param_init 0 -param_init_glorot -max_generator_batches 32 \
	-batch_size 128 -valid_batch_size 8 -batch_type tokens -normalization tokens -max_grad_norm 0 \
	-accum_count 4 -optim adam -adam_beta1 0.9 -adam_beta2 0.998 -decay_method noam \
	-warmup_steps 2000 -learning_rate 2 -label_smoothing 0.05 -report_every 1000 \
	-layers 4 -rnn_size 256 -word_vec_size 256 -encoder_type transformer \
	-decoder_type transformer -dropout 0.1 -position_encoding -share_embeddings \
	-global_attention general -global_attention_function softmax \
	-self_attn_type scaled-dot -heads 8 -transformer_ff 2048 \
	-train_from ${save_model}toy_model_step_50000.pt \
