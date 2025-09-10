from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from transformers import GenerationConfig
import torch
import os.path as op
import os
from PIL import Image
import random
import gc
from iotools import read_json
import json
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import requests
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

datasets_path = '/home/yangxiaohui/SCL/datasets/ICFG-PEDES/'
image_path = '/home/yangxiaohui/SCL/datasets/ICFG-PEDES/'
anno_path = op.join(datasets_path, 'ICFG-PEDES.json')
train_image_path = []

annos = read_json(anno_path)
for anno in annos:
    if anno['split'] == 'train':
        train_image_path.append(anno['file_path'])

iter = len(train_image_path)

model_path = '/home/yangxiaohui/SCL/Qwen2-VL-7B/'
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_path, torch_dtype=torch.float16, device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_path)

templates = [
    "Wearing [clothing description], the [person/woman/man] also has [hair description] and is carrying [belongings description]. The person is [figure description].",
    "Sporting [hair description], the [person/woman/man] is dressed in [clothing description] and is carrying [belongings description].",
    "With [hair description], the [person/woman/man] is wearing [clothing description] and is also carrying [belongings description].",
    "In [clothing description] and [footwear description], the [person/woman/man] is also carrying [belongings description].",
    "Carrying [belongings description], the [person/woman/man] is dressed in [clothing description] and [footwear description].",
    "In [clothing description] and [footwear description], the [person/woman/man] also has [hair description].",
    "Carrying [belongings description], the [person/woman/man] is wearing [clothing description] and [footwear description].",
    "With [hair description], the [person/woman/man] is dressed in [clothing description] and [accessory description].",
    "With [footwear description], the [person/woman/man] is wearing [clothing description] and [accessory description].",
    "With [hair description], the [person/woman/man] is wearing [clothing description] with [accessory description].",
    "In [clothing description] and [accessory description], the [person/woman/man] also has [hair description].",
    "With [accessory description], the [person/woman/man] also has [hair description] and is carrying [belongings description].",
    "Wearing [clothing description] and [footwear description], the [person/woman/man] also has [hair description].",
    "The [person/woman/man] is wearing [footwear description], [accessory description], [clothing description], and [belongings description]. The [person/woman/man] has [hair description].",
    "The [person/woman/man] has [hair description] and is wearing [accessory description], [footwear description], [clothing description], and carrying [belongings description].",
    "The [person/woman/man] is dressed in [footwear description], [clothing description], [accessory description], and carrying [belongings description]. The [person/woman/man] has [hair description].",
    "With [footwear description], the [person/woman/man] is wearing [clothing description], [accessory description], and carrying [belongings description]. The [person/woman/man] has [hair description].",
    "The [person/woman/man] sports [hair description] and is dressed in [footwear description], [clothing description], [accessory description], and carrying [belongings description].",
    "Wearing [footwear description], [accessory description], [clothing description], the [person/woman/man] is also carrying [belongings description]. The [person/woman/man] has [hair description].",
    "The [person/woman/man] is attired in [clothing description], [accessory description], [footwear description], and carrying [belongings description]. The [person/woman/man] has [hair description].",
    "The [person/woman/man] is seen wearing [footwear description], [clothing description], [accessory description], and carrying [belongings description]. The [person/woman/man] has [hair description].",
    "With [hair description], the [person/woman/man] is wearing [footwear description], [clothing description], [accessory description], and carrying [belongings description].",
    "Dressed in [footwear description], [accessory description], [clothing description], and carrying [belongings description], the [person/woman/man] has [hair description].",
    "The [person/woman/man] can be seen wearing [footwear description], [clothing description], [accessory description], and carrying [belongings description]. The [person/woman/man] has [hair description].",
    "The [person/woman/man] is dressed in [clothing description], [footwear description], [accessory description], and carrying [belongings description]. The [person/woman/man] has [hair description].",
    "The [person/woman/man] is wearing [footwear description], [accessory description], [clothing description], and carrying [belongings description]. The [person/woman/man] has [hair description].",
    "The [person/woman/man] is attired in [accessory description], [footwear description], [clothing description], and carrying [belongings description]. The [person/woman/man] has [hair description].",
    "The [person/woman/man] has [hair description] and is wearing [clothing description], [footwear description], [accessory description], and carrying [belongings description].",
    "In [accessory description], [footwear description], [clothing description], and carrying [belongings description], the [person/woman/man] has [hair description].",
    "The [person/woman/man] is seen wearing [clothing description], [footwear description], [accessory description], and carrying [belongings description]. The [person/woman/man] has [hair description].",
    "The [person/woman/man] is wearing [accessory description], [footwear description], [clothing description], and carrying [belongings description]. The [person/woman/man] has [hair description].",
    "The [person/woman/man] is seen in [footwear description], [accessory description], [clothing description], and carrying [belongings description]. The [person/woman/man] has [hair description].",
    "Carrying [belongings description], the [person/woman/man] is stylishly outfitted in [clothing description], complemented by [accessory description], and stepping out in [footwear description]. They have [hair description].",
    "The [person/woman/man] can be spotted wearing [accessory description], [footwear description], [clothing description], and carrying [belongings description]. The [person/woman/man] has [hair description].",
    "The [person/woman/man] has [hair description] and is dressed in [accessory description], [footwear description], [clothing description], and carrying [belongings description].",
    "The [person/woman/man] is attired in [accessory description], [clothing description], [footwear description], and carrying [belongings description]. The [person/woman/man] has [hair description].",
    "The [person/woman/man] is wearing [accessory description], [clothing description], [footwear description], and carrying [belongings description]. The [person/woman/man] has [hair description].",
    "With [hair description], the [person/woman/man] is wearing [accessory description], [clothing description], [footwear description], and carrying [belongings description].",
    "Dressed in [accessory description], [clothing description], [footwear description], and carrying [belongings description], the [person/woman/man] has [hair description].",
    "The [person/woman/man] can be seen wearing [accessory description], [clothing description], [footwear description], and carrying [belongings description]. The [person/woman/man] has [hair description].",
    "The [person/woman/man] is dressed in [clothing description], [accessory description], [footwear description], and carrying [belongings description]. The [person/woman/man] has [hair description].",
    "The [person/woman/man] is wearing [clothing description], [accessory description], [footwear description], and carrying [belongings description]. The [person/woman/man] has [hair description]."
]
att = ['clothing','shoes','hairstyle','gender','belongings']
random.seed(41)
data_list = []
progress_bar = tqdm(range(iter), desc="Processing images")
for i in progress_bar:
    random_number = random.randint(0, 41)
    url = image_path + train_image_path[i]
    image = Image.open(url)
    factor = 28
    original_width, original_height = image.size
    if original_width < factor or original_height < factor:
        new_size = (60, 160)
    else:
        new_size = (original_width, original_height)

    resized_image = image.resize(new_size, Image.BICUBIC)

    conversation = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                },
                #{"type": "text", "text": f'According to the template: {templates[random_number]}, generate a description about the overall appearance of the person, including the {att[0]}, {att[1]}, {att[2]}, {att[3]} ,{att[4]} and posture. The sentence pattern and template must be consistent. If some requirements in the template are not visible, you can ignore. Do not imagine any contents that are not in the image.'},
                {"type": "text", "text": f"Describe this image using the following template: {templates[random_number]}"}
            ],
        }
    ]
    text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(
        text=[text_prompt], images=[resized_image], padding=True, return_tensors="pt"
    )
    inputs = inputs.to("cuda").to(torch.float16)
    output_ids = model.generate(**inputs, max_new_tokens=128, temperature=0.7, top_p=0.9)
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(inputs.input_ids, output_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    data_list.append({train_image_path[i]: output_text})
    progress_bar.set_postfix_str(f"Processed {i + 1} out of {iter}")
    progress_bar.update(1)

with open('qwen2-vl-icfg.json', 'w') as f:
    json.dump(data_list, f, indent=4)
print("Processing complete.")

