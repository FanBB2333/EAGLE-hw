from lmms_eval.tasks import TaskManager, get_task_dict

task_name = "librispeech_dev_clean"

task_manager = TaskManager("INFO", model_name="eagle")
task_dict = get_task_dict([task_name], task_manager)
dataset = task_dict[task_name].dataset[task_name]
