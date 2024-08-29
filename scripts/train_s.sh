torchrun --nnodes=1 --nproc_per_node=8 train.py \
--version small \
--data-path combine_dataset.json \
--global_batch_size 128 \
--global-seed 2023