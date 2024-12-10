export CUDA_VISIBLE_DEVICES=1

model_name=/home/caicai/vp/AlphaViP-LLaVA/checkpoints/alpha-vip-llava-7b-vision_tower-2

python llava/eval/model_vqa_loader_vip.py  \
      --model-path  $model_name  \
      --question-file ./playground/data/eval/v7w-test.json \
      --image-folder  ./playground/data \
      --alpha 128 \
      --answers-file ./playground/data/eval/v7w-test-answer.json
