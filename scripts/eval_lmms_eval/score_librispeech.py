import json
from tqdm import tqdm
from jiwer import wer
from lmms_eval.tasks import TaskManager, get_task_dict

task_name = "librispeech_dev_clean"
split_name = "librispeech_dev_clean"
result_file = "/home1/hxl/disk/EAGLE/output/eval/20250614/('librispeech_dev_clean',).json"

task_manager = TaskManager("INFO", model_name="eagle")
task_dict = get_task_dict([task_name], task_manager)
dataset = task_dict[task_name].dataset[split_name]

results = json.load(open(result_file, "r"))
result_dict = {int(item["doc_id"]): item for item in results}

score_results = []
preds, gts = [], []
for i, ann in enumerate(tqdm(dataset)):
    gt = ann["gt"]
    gts.append(gt)
    if i not in result_dict:
        print(f"Warning: No result for index {i}")
        preds.append("")
        continue
    result_item = result_dict[i]
    result_item["gt"] = gt
    score_results.append(result_item)
    pred = result_item["text_output"]
    preds.append(pred)


print(f"WER: {wer(gts, preds):.2f}%")
with open(result_file.replace(".json", "_with_gt.json"), "w") as f:
    json.dump(score_results, f, indent=4)
