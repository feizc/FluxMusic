torchrun --nnodes=4 --nproc_per_node=8 train.py \
--version giant \
--data-path combine_dataset.json \
--resume results/giant/checkpoints/0050000.pt \
--global_batch_size 128 \
--global-seed 2023 \
--accum_iter 8
