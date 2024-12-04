from transformers import GenerationConfig

# 创建一个 GenerationConfig 对象
gen_config = GenerationConfig(
    max_length=50,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    num_return_sequences=3,
    pad_token_id=0,
    bos_token_id=1,
    eos_token_id=2,
)

# 保存到指定路径
gen_config.save_pretrained("./config")

# 文件会保存为 ./config/generation_config.json


# from datasets import load_dataset

# dataset_name = "imdb"  # 替換成您想下載的數據集名稱
# dataset = load_dataset(dataset_name)

# # 儲存到本地
# dataset.save_to_disk("./local_dataset")





# from PIL import Image
# import ipdb
# import torch
# from transformers import AutoProcessor, VipLlavaForConditionalGeneration

# from llava.model import LlavaLlamaForCausalLM, LlavaConfig






# model = LlavaLlamaForCausalLM.from_pretrained(
#     model_args.model_name_or_path,
#     cache_dir=training_args.cache_dir,
#     attn_implementation=attn_implementation,
#     torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
#     **bnb_model_from_pretrained_args
# )















# from PIL import Image
# import ipdb
# import torch
# from transformers import AutoProcessor, VipLlavaForConditionalGeneration

# model_id = "llava-hf/vip-llava-7b-hf"
# model = VipLlavaForConditionalGeneration.from_pretrained(
#     model_id,
#     torch_dtype=torch.float16, 
#     low_cpu_mem_usage=True, 
# ).to(0)

# processor = AutoProcessor.from_pretrained(model_id)

# # Define a chat histiry and use `apply_chat_template` to get correctly formatted prompt
# # Each value in "content" has to be a list of dicts with types ("text", "image") 
# conversation = [
#     {

#       "role": "user",
#       "content": [
#           {"type": "text", "text": "What are these?"},
#           {"type": "image"},
#         ],
#     },
# ]
# prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

# image_file = "/home/caicai/vp/AlphaViP-LLaVA/000000039769.jpg"
# raw_image = Image.open(image_file)
# inputs = processor(images=raw_image, text=prompt, return_tensors='pt').to(0, torch.float16)

# # ipdb.set_trace()

# output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
# print(processor.decode(output[0][2:], skip_special_tokens=True))
