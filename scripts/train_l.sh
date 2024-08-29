torchrun --nnodes=4 --nproc_per_node=8 train.py \
--version large \
--data-path combine_dataset.json \
--global_batch_size 128 \
--global-seed 2023