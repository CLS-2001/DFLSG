import json
import tempfile
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap


with open('/home/yangxiaohui/SCL/datasets/CUHK-PEDES/reid_raw.json', 'r') as f:
    reid_data = json.load(f)

with open('/home/yangxiaohui/SCL/datasets/CUHK-PEDES/GTR.json', 'r') as f:
    transcp_data = json.load(f)

with open('/home/yangxiaohui/SCL/datasets/CUHK-PEDES/qwen1w2-cuhk.json', 'r') as f:
    qwen_data = json.load(f)

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

def format_for_coco(pred_dict, ref_dict):
    preds = []
    refs = {}
    for img_id, pred in pred_dict.items():
        preds.append({"image_id": img_id, "caption": pred})
        refs[img_id] = ref_dict.get(img_id, [])
    return preds, refs

def run_cider(pred_dict, ref_dict):
    preds, refs = format_for_coco(pred_dict, ref_dict)

    with tempfile.NamedTemporaryFile('w+', suffix='.json', delete=False) as pred_file, \
         tempfile.NamedTemporaryFile('w+', suffix='.json', delete=False) as ref_file:
        json.dump(preds, pred_file)
        json.dump(
            {
                "annotations": [
                    {"image_id": img_id, "id": idx, "caption": cap}
                    for idx, (img_id, caps) in enumerate(refs.items())
                    for cap in caps
                ],
                "images": [{"id": img_id} for img_id in refs.keys()],
                "type": "captions",
                "info": "dummy"
            },
            ref_file
        )
        pred_file.flush()
        ref_file.flush()

        coco = COCO(ref_file.name)
        cocoRes = coco.loadRes(pred_file.name)

        cocoEval = COCOEvalCap(coco, cocoRes)
        cocoEval.evaluate()

        return cocoEval.eval['CIDEr']


common_keys = set(gt_captions) & (set(qwen_captions) | set(transcp_captions))
gt_filtered = {k: [gt_captions[k]] for k in common_keys}
qwen_filtered = {k: v for k, v in qwen_captions.items() if k in common_keys}
transcp_filtered = {k: v for k, v in transcp_captions.items() if k in common_keys}

cider_qwen = run_cider(qwen_filtered, gt_filtered)
cider_transcp = run_cider(transcp_filtered, gt_filtered)

print(f"Qwen     CIDEr: {cider_qwen:.4f}")
print(f"TransCP  CIDEr: {cider_transcp:.4f}")
