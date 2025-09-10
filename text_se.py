from model import build_model
from utils.options import get_args
import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import json
from utils.iotools import read_image
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
from utils.simple_tokenizer import SimpleTokenizer
import torch.nn.functional as F
from utils.checkpoint import Checkpointer
from tqdm import tqdm
args = get_args()

model = build_model(args)
checkpointer = Checkpointer(model)
checkpointer.load("/home/yangxiaohui/SCL/CUHK-PEDES/best.pth")
model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

tokenizer = SimpleTokenizer()

def tokenize(caption: str, tokenizer, text_length=77, truncate=True) -> torch.LongTensor:
    sot_token = tokenizer.encoder["<|startoftext|>"]
    eot_token = tokenizer.encoder["<|endoftext|>"]
    tokens = [sot_token] + tokenizer.encode(caption) + [eot_token]
    result = torch.zeros(text_length, dtype=torch.long)
    if len(tokens) > text_length:
        if truncate:
            tokens = tokens[:text_length]
            tokens[-1] = eot_token
        else:
            raise RuntimeError(
                f"Input {caption} is too long for context length {text_length}"
            )
    result[:len(tokens)] = torch.tensor(tokens)
    return result


def extract_features(image, text1, text2, model):
    with torch.no_grad():
        image_features = model.encode_image(image)
        text1_features = model.encode_text(text1)
        text2_features = model.encode_text(text2)
    return image_features, text1_features, text2_features

def cosine_similarity(features1, features2):
    return F.cosine_similarity(features1, features2)

preprocess = transforms.Compose([
    transforms.Resize((384, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
])

with open('/home/yangxiaohui/SCL/datasets/ICFG-PEDES/ICFG_LLAVA.json', 'r', encoding='utf-8') as f1, open('/home/yangxiaohui/SCL/datasets/ICFG-PEDES/ICFG_deep.json', 'r', encoding='utf-8') as f2:
    data1 = json.load(f1)
    data2 = json.load(f2)

for i in tqdm(range(len(data1)), desc="Processing Images"):
    key1 = list(data1[i].keys())[0]
    text1 = data1[i][key1]
    text2 = list(data2[i].values())[0]

    url = '/home/yangxiaohui/SCL/datasets/ICFG-PEDES/imgs/'+ key1

    img = read_image(url)
    img = preprocess(img).to(device)
    img = img.unsqueeze(0)
    #img = img.to(torch.float16)

    tokens1 = tokenize(text1, tokenizer=tokenizer, text_length=77, truncate=True).to(device).unsqueeze(0)
    tokens2 = tokenize(text2, tokenizer=tokenizer, text_length=77, truncate=True).to(device).unsqueeze(0)

    image_features,text1_features,text2_features = extract_features(img,tokens1,tokens2, model)

    similarity1 = cosine_similarity(image_features, text1_features)
    similarity2 = cosine_similarity(image_features, text2_features)

    if similarity2 > similarity1:
        data1[i][key1] = text2


with open('LLAVA&DEEP-ICFG_review.json', 'w', encoding='utf-8') as f1_updated:
    json.dump(data1, f1_updated, ensure_ascii=False, indent=4)

print("update completed")