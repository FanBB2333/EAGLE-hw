# import sys
# sys.path.append('.')
# from eagle.model.multimodal_encoder.languagebind.video.processing_video import load_and_transform_video

# from torchvision.transforms import Compose, Lambda, ToTensor
# from torchvision.transforms._transforms_video import NormalizeVideo, RandomCropVideo, RandomHorizontalFlipVideo, CenterCropVideo
# from pytorchvideo.transforms import ApplyTransformToKey, ShortSideScale, UniformTemporalSubsample

# from tqdm import tqdm
# import pyarrow as pa
# import pyarrow.ipc as ipc
# import os

# video_path = '.cache/huggingface/YouCookIIVideos/YouCookIIVideos/val/_GTwKEPmB-U_0.mp4'

# OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
# OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)

# transform_torch = ApplyTransformToKey(
#     key="video",
#     transform=Compose(
#         [
#             UniformTemporalSubsample(8),
#             Lambda(lambda x: x / 255.0),
#             NormalizeVideo(mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD),
#             ShortSideScale(size=224),
#             CenterCropVideo(224),
#             RandomHorizontalFlipVideo(p=0.5),
#         ]
#     ),
# )
# try:
#     feature = load_and_transform_video(
#         video_path=video_path,
#         transform=transform_torch,
#         video_decode_backend='pytorchvideo'
#     )['video']
#     print(feature.shape)
# except Exception as e:
#     print(e)
    
# import torch
# torch.save(feature, 'try/try.pt')

# import numpy as np

# np.save('try/try.npy', feature)
from tqdm import tqdm
import torch
# for i in tqdm(range(100000)):
#     feature = torch.load('try/try.pt')

# import numpy as np

# for i in tqdm(range(100000)):
#     feature = torch.from_numpy(np.load('try/try.npy'))

model = torch.load("/home1/hxl/disk2/MLLM/ModelCompose/checkpoints/train-vision-pr-llm/adapter_model.bin")

print(model)