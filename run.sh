export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

python3 eval.py
