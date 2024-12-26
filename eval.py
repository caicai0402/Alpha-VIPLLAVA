import numpy as np
from PIL import Image
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model


def generate_rgb_diff(image1, image2):
    array1 = np.array(image1.convert("RGB"))
    array2 = np.array(image2.convert("RGB"))
    diff_mask = np.any(array1 != array2, axis=-1).astype(np.uint8)
    return Image.fromarray(diff_mask * 255)

model_path = "/data2/caicai/AlphaVip-LLaVA/checkpoints/alpha-vip-llava-7b-2"
image_file = "/data2/caicai/AlphaVip-LLaVA/playground/data/example/img_vp1.jpg"
alpha_image_file = "/data2/caicai/AlphaVip-LLaVA/playground/data/example/alpha1.jpg"
prompt = "What is shown within the lighlighted region?"

args = type('Args', (), {
    "model_path": model_path,
    "model_name": get_model_name_from_path(model_path),
    "query": prompt,
    "image_file": image_file,
    "alpha_image_file": alpha_image_file,
    "conv_mode": None, "model_base": None, "temperature": 0.2, "top_p": None, "num_beams": 1, "max_new_tokens": 512, "sep": ",",
})()

eval_model(args)
