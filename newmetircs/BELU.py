import json
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

with open('/home/yangxiaohui/SCL/datasets/CUHK-PEDES/GTR.json', 'r') as f:
    transcp_data = json.load(f)

with open('/home/yangxiaohui/SCL/datasets/CUHK-PEDES/qwen1w2-cuhk.json', 'r') as f:
    qwen_data = json.load(f)

with open('/home/yangxiaohui/SCL/datasets/CUHK-PEDES/reid_raw.json', 'r') as f:
    reid_data = json.load(f)

gt_captions = {}
for item in reid_data:
    key = item['file_path'].replace('\\', '/')
    if len(item['captions']) > 0:
        gt_captions[key] = item['captions'][0]

transcp_captions = {}
for item in transcp_data:
    key = item['file_path'].replace('\\', '/')
    transcp_captions[key] = item['captions'][0]

qwen_captions = {}
for item in qwen_data:
    for k, v in item.items():
        key = k.replace('\\', '/')
        qwen_captions[key] = v

smoothie = SmoothingFunction().method4
def compute_bleus(reference, hypothesis):
    ref = [reference.lower().split()]
    hyp = hypothesis.lower().split()
    bleu1 = sentence_bleu(ref, hyp, weights=(1, 0, 0, 0), smoothing_function=smoothie)
    bleu2 = sentence_bleu(ref, hyp, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie)
    bleu3 = sentence_bleu(ref, hyp, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothie)
    bleu4 = sentence_bleu(ref, hyp, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)
    return bleu1, bleu2, bleu3, bleu4, (bleu1 + bleu2 + bleu3 + bleu4) / 4

common_keys = set(gt_captions.keys()) & (set(transcp_captions.keys()) | set(qwen_captions.keys()))

transcp_bleus = []
qwen_bleus = []

for key in common_keys:
    ref = gt_captions[key]
    if key in transcp_captions:
        transcp_bleus.append(compute_bleus(ref, transcp_captions[key]))
    if key in qwen_captions:
        qwen_bleus.append(compute_bleus(ref, qwen_captions[key]))

def avg_bleu(bleu_list):
    if not bleu_list:
        return [0]*5
    b1 = sum(b[0] for b in bleu_list) / len(bleu_list)
    b2 = sum(b[1] for b in bleu_list) / len(bleu_list)
    b3 = sum(b[2] for b in bleu_list) / len(bleu_list)
    b4 = sum(b[3] for b in bleu_list) / len(bleu_list)
    avg = sum(b[4] for b in bleu_list) / len(bleu_list)
    return [b1, b2, b3, b4, avg]

b1, b2, b3, b4, avg_transcp = avg_bleu(transcp_bleus)
print(f"TransCP BLEU-1: {b1:.4f}, BLEU-2: {b2:.4f}, BLEU-3: {b3:.4f}, BLEU-4: {b4:.4f}, AVG: {avg_transcp:.4f}")

b1, b2, b3, b4, avg_qwen = avg_bleu(qwen_bleus)
print(f"Qwen    BLEU-1: {b1:.4f}, BLEU-2: {b2:.4f}, BLEU-3: {b3:.4f}, BLEU-4: {b4:.4f}, AVG: {avg_qwen:.4f}")
