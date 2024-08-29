torchrun --nnodes=2 --nproc_per_node=8 train.py \
--version base \
--data-path combine_dataset.json \
--global_batch_size 128 \
--resume results/base/checkpoints/0050000.pt \
--global-seed 2023