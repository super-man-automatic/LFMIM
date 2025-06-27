CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 LFMIMv2.py \
	    --rank 0 \
        --world_size 1 \
        --init_method env:// 
