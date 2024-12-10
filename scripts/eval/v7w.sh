export CUDA_VISIBLE_DEVICES=6

model_name=checkpoints/alpha-vip-llava-7b-2-caicai

python llava/eval/model_vqa_loader_vip.py  \
      --model-path  $model_name  \
      --question-file ./playground/data/eval/v7w-test.json \
      --image-folder  ./playground/data \
      --alpha 128 \
      --answers-file ./playground/data/eval/v7w-test-answer.json
