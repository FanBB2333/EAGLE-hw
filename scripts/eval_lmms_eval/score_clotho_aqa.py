import json
from tqdm import tqdm
from lmms_eval.tasks import TaskManager, get_task_dict

task_name = "clotho_aqa_test"
split_name = "clotho_aqa_test_filtered"
result_file = "/home1/hxl/disk/EAGLE/output/eval/20250614/('clotho_aqa_test',).json"

task_manager = TaskManager("INFO", model_name="eagle")
task_dict = get_task_dict([task_name], task_manager)
dataset = task_dict[task_name].dataset[split_name]

results = json.load(open(result_file, "r"))
result_dict = {int(item["doc_id"]): item for item in results}

correct = 0
score_results = []
for i, ann in enumerate(tqdm(dataset)):
    gt = ann["answer"]
    if i not in result_dict:
        print(f"Warning: No result for index {i}")
        continue
    result_item = result_dict[i]
    result_item["gt"] = gt
    score_results.append(result_item)
    pred = result_item["text_output"]
    if pred == gt:
        correct += 1

accuracy = correct / len(dataset) * 100
print(f"Accuracy: {accuracy:.2f}%")
with open(result_file.replace(".json", "_with_gt.json"), "w") as f:
    json.dump(score_results, f, indent=4)