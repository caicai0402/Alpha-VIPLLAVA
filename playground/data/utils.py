import json

def extract_coco_from_mix(input_path, output_path):
    with open(input_path, "r") as f:
        data = json.load(f)
    coco_data = []
    for d in data:
        if "image" in d.keys() and "coco" in d["image"]:
            coco_data.append(d)
    with open(output_path, "w") as f:
        json.dump(coco_data, f, indent=4)

def fun(json_file):
    with open(json_file, "r") as f:
        data = json.load(f)
    
    count = 0
    for d in data:
        if "bboxes" not in d.keys():
            print(d["id"])

        elif len(d["bboxes"]) > 1:
            count += 1

    print(count)

if __name__ == "__main__":
    # extract_coco_from_mix(
    #     "/home/caicai/vp/AlphaViP-LLaVA/playground/data/vip-llava_stage3_mix.json",
    #     "/home/caicai/vp/AlphaViP-LLaVA/playground/data/vip-llava_stage3_coco.json"
    # )

    fun("/home/caicai/vp/AlphaViP-LLaVA/playground/data/vip-llava_stage3_coco.json")
