from pathlib import Path
from datasets import load_dataset
import os
import json
from tqdm import tqdm

root_dir = Path("download")
ds_name = "remyxai/OpenSpaces"
ds = load_dataset(ds_name)
ds_name = ds_name.split('/')[-1]

ds_local_path = root_dir / ds_name

os.makedirs(ds_local_path, exist_ok=True)
images_dir = ds_local_path / "images"
os.makedirs(images_dir, exist_ok=True)

llava_format_json = []

role_map = {
    "assistant": "gpt",
    "user": "human"
}

img_cnt = 0
for _, ds_split in ds.items():
    for ds_el in tqdm(ds_split):
        conv_id = f"{img_cnt:08}"
        img_cnt += 1
        image = ds_el["images"][0]
        local_image_path = f"images/{conv_id}.jpg"
        image.save(ds_local_path / local_image_path)
        src_messages = ds_el["messages"]
        dst_ds_el = {
            "id": conv_id,
            "image": local_image_path,
            "conversations": []
        }
        dst_conversations = dst_ds_el["conversations"]

        for message in src_messages:
            content = message["content"]
            dst_value = []
            for c_el in content:
                dst_value.append("<image>" if c_el["type"] == "image" else c_el["text"])

            dst_message = {
                "from": role_map[message["role"]],
                "value": "\n".join(dst_value)
            }
            dst_conversations.append(dst_message)

        llava_format_json.append(dst_ds_el)

json_file = open(ds_local_path / (ds_name + ".json"), 'w')
json_file.write(json.dumps(llava_format_json))
