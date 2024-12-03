from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model

model_path = "/home/caicai/vp/AlphaViP-LLaVA/checkpoints/vip-llava-7b"
prompt = "What is shown within the pointed region?"
image_file = "./man-cross-street.jpg"

args = type('Args', (), {
    "model_path": model_path,
    "model_name": get_model_name_from_path(model_path),
    "query": prompt,
    "image_file": image_file,
    "conv_mode": None, "model_base": None, "temperature": 0.2, "top_p": None, "num_beams": 1, "max_new_tokens": 512, "sep": ",",
})()

eval_model(args)
