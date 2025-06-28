import datetime
import json
import os
from collections import defaultdict

from loguru import logger as eval_logger

# from lmms_eval.tasks._task_utils.file_utils import generate_submission_file

dir_name = os.path.dirname(os.path.abspath(__file__))

eval_type_dict = {
    "Perception": [
        "existence",
        "count",
        "position",
        "color",
        "posters",
        "celebrity",
        "scene",
        "landmark",
        "artwork",
        "OCR",
    ],
    "Cognition": [
        "commonsense_reasoning",
        "numerical_calculation",
        "text_translation",
        "code_reasoning",
    ],
}


replace_prompt = " Please answer yes or no."


def mme_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def mme_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["question"].strip()
    if "pre_prompt" in lmms_eval_specific_kwargs and lmms_eval_specific_kwargs["pre_prompt"] != "":
        question = question.replace(replace_prompt, "")
        question = f"{lmms_eval_specific_kwargs['pre_prompt']}{question}"
    if "post_prompt" in lmms_eval_specific_kwargs and lmms_eval_specific_kwargs["post_prompt"] != "":
        question = question.replace(replace_prompt, "")
        question = f"{question}{lmms_eval_specific_kwargs['post_prompt']}"
    return question


def parse_pred_ans(pred_ans):
    """Brought from Otter Eval"""
    pred_ans = pred_ans.lower().strip().replace(".", "")
    pred_label = None
    if pred_ans in ["yes", "no"]:
        pred_label = pred_ans
    elif len(pred_ans) == 1:
        if pred_ans == "y":
            pred_label = "yes"
        elif pred_ans == "n":
            pred_label = "no"
        else:
            pred_label = "other"
    else:
        prefix_pred_ans = pred_ans[:4]
        if "yes" in prefix_pred_ans:
            pred_label = "yes"
        elif "no" in prefix_pred_ans:
            pred_label = "no"
        else:
            pred_label = "other"
    return pred_label


def mme_process_results(doc, results):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name (in this case mme score), value: metric value
    """
    pred = results[0]
    pred_ans = parse_pred_ans(pred)
    gt_ans = doc["answer"].lower().strip().replace(".", "")
    assert gt_ans in ["yes", "no"]
    assert pred_ans in ["yes", "no", "other"]
    score = 1.0 if pred_ans == gt_ans else 0.0
    category = doc["category"]
    key_name = "mme_perception_score" if category in eval_type_dict["Perception"] else "mme_cognition_score"
    # Note: the key name here is very important. It decides which aggregation function will receive the results
    # We note down the question id/category to help us aggregate the results later
    return {key_name: {"question_id": doc["question_id"], "category": category, "score": score}}


def mme_aggregate_results(results):
    """
    Args:
        results: a list of values returned by process_results
    Returns:
        A score
    """
    category2score = defaultdict(dict)
    for result in results:
        question_id = result["question_id"]
        score = result["score"]
        category = result["category"]
        if question_id not in category2score[category]:
            category2score[category][question_id] = []
        category2score[category][question_id].append(score)
    category2avg_score = {}
    # print(category2score)
    for category, question2scores in category2score.items():
        total_score = 0
        for question_id, scores in question2scores.items():
            # BEGIN hxl
            if len(scores) == 1:
                print(f'Lost question {question_id}')
                scores.append(0.0)
            # END hxl
            assert len(scores) == 2, f"MME only supports pairwise evaluation, scores: {scores}"
            acc = sum(scores) / len(scores) * 100.0
            acc_plus = (sum(scores) == 2) * 100.0
            score = acc_plus + acc
            total_score += score
        avg_score = total_score / len(question2scores)
        category2avg_score[category] = avg_score
    for category, avg_score in category2avg_score.items():
        eval_logger.info(f"{category}: {avg_score:.2f}")
    total_score = sum(category2avg_score.values())
    return total_score


def eval_score():
    from datasets import load_dataset
    ds = load_dataset("lmms-lab/MME")
    ds = ds['test']
    #     ds[0]: 
    # {'question_id': 'code_reasoning/0020.png',
    #  'image': <PIL.PngImagePlugin.PngImageFile image mode=RGB size=416x114>,
    #  'question': 'Is a c++ code shown in the picture? Please answer yes or no.',
    #  'answer': 'No',
    #  'category': 'code_reasoning'}

    results_path = "/home1/hxl/disk/EAGLE/output/eval/20250518/('mme',).json"
    with open(results_path, "r") as f:
        preds = json.load(f)
    # sample result path:
    # {
    #     "text_output": "Yes",
    #     "context": "Is this a photo of Torre del Rellotge d'Olesa de Montserrat?\nAnswer the question using a single word or phrase.",
    #     "doc_id": 1526
    # },
    
    # filter the questions according to eval_type_dict
    # use mme_aggregate_results function, to calculate the score
    # fill in the following calculation process
        docid2pred = {item["doc_id"]: item["text_output"] for item in preds}

    # 仅保留在 Perception 和 Cognition 中定义的类别的问题
    valid_categories = set(eval_type_dict["Perception"] + eval_type_dict["Cognition"])
    filtered_docs = [doc for doc in ds if doc["category"] in valid_categories]

    results = []
    lost_count = 0

    # 对每个样本进行处理和评分
    for doc in filtered_docs:
        doc_id = doc["question_id"]  # e.g., code_reasoning/0020.png
        # 将 question_id 转换为 doc_id 的整数编号
        try:
            numeric_id = int(doc_id.split("/")[-1].split(".")[0])
        except ValueError:
            eval_logger.warning(f"Invalid question_id format: {doc_id}")
            continue

        if numeric_id not in docid2pred:
            eval_logger.warning(f"Missing prediction for doc_id: {numeric_id}")
            lost_count += 1
            continue

        pred_text = docid2pred[numeric_id]
        result = mme_process_results(doc, [pred_text])
        # result 是一个字典 {'mme_cognition_score': {...}} 或 {'mme_perception_score': {...}}
        results.extend(result.values())

    if lost_count > 0:
        eval_logger.warning(f"Total {lost_count} predictions missing from result file.")

    # 聚合并返回总分
    final_score = mme_aggregate_results(results)
    return final_score

if __name__ == '__main__':
    results = eval_score()
    print(results)
    
    
    