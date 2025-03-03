import os
os.environ['TRANSFORMERS_CACHE'] = '/archive2/arr159/huggingface/'
import pandas as pd
from transformers import CLIPProcessor,CLIPImageProcessor, CLIPModel#, get_image_features
from PIL import Image
import torch
import torchvision
from torchvision.io import read_video
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
from pathlib import Path
def extract_clips(video_path, clip_length=32):
    """
    Extracts non-overlapping clips of a given length from a video.

    :param video_path: Path to the video file.
    :param clip_length: Length of each clip in frames.
    :return: A list of clips, each clip is a tensor of shape (clip_length, C, H, W).
    """
    # Read the video
    # vr = torchvision.io.VideoReader(video_path, "video")
    video_frames, _, _ = torchvision.io.read_video(str(video_path), pts_unit='sec')

    clips = []
    num_frames = video_frames.shape[0]

    for start in range(0, num_frames, clip_length):
        end = start + clip_length
        if end <= num_frames:
            # Extract the clip and adjust dimensions to (clip_length, C, H, W)
            clip = video_frames[start:end].permute(0, 3, 1, 2)
        else:
            clip = video_frames[start:].permute(0, 3, 1, 2)
        clips.append(clip)
    # breakpoint()
    return clips


def sample_tensor(tensor, fn):
    """
    Samples a tensor based on the specified function argument.

    :param tensor: A tensor (e.g., a numpy array) representing a sequence.
    :param fn: A string that specifies the sampling strategy.
               Can be 'start', 'middle', 'end', or 'random'.
    :return: An element from the tensor.
    """
    length = len(tensor)

    if fn == 'start':
        # Sample from the start of the tensor
        return tensor[0]
    elif fn == 'middle':
        # Sample from the middle of the tensor
        middle_index = length // 2
        return tensor[middle_index]
    elif fn == 'end':
        # Sample from the end of the tensor
        return tensor[-1]
    elif fn == 'random':
        # Sample a random element from the tensor
        random_index = random.randint(0, length - 1)
        return tensor[random_index]
    elif fn == 'all':
        return tensor
    elif fn == 'random_five':
        tensors = []
        if tensor.shape[0] <= 5:
            return tensor
        else:
            # random sample
            for i in random.sample(range(length-1), 5):
                tensors.append(tensor[i])

        return torch.stack(tensors)
    else:
        raise ValueError("Invalid sampling function. Choose 'start', 'middle', 'end', or 'random'.")


# Create the parser
parser = argparse.ArgumentParser(description="A simple argparse example")

# Add arguments
# parser.add_argument('--sample_position', type=str, default="middle")
parser.add_argument('--vlm_type', type=str, default="clip")
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--partition', type=int, default=0)
parser.add_argument('--original', action='store_true', default=False)
parser.add_argument('--save_path', type=str, default=Path('/archive1/arr159/gdlt'))
parser.add_argument('--video_path', type=Path, default=Path('/halcyon/archive1/arr159/fis-v/videos/'))
parser.add_argument('--data_frame', type=Path, default=Path('/afs/cs.pitt.edu/usr0/arr159/figure_skating/MS_LSTM/data/fs_dataset.csv'))
args = parser.parse_args()

model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(args.device)
processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
video_list = pd.read_csv(args.data_frame)['number'].values
if args.partition != -1:
    partition_size = len(video_list)//10 + 10 # eliminate scope of remainders
    video_list = video_list[partition_size*args.partition:partition_size*(args.partition+1)+1]
else:
    video_list = [video_name for video_name in video_list if not (args.save_path / args.vlm_type / 'random_five' / str(video_name)).with_suffix('.npy').exists()]
    print("Processing Videos:", len(video_list))

from tqdm import tqdm
for video_name in tqdm(video_list):
    video_path = (args.video_path / str(video_name)).with_suffix('.mp4')
    if not video_path.exists():
        video_path = video_path.with_suffix('.mov')
        if not video_path.exists():
            print("Video Path DNE:", video_path)
            continue
    try:
        clips = extract_clips(video_path)
        if args.original:
            for sample_position in ['start', 'end', 'middle', 'random']:
                save_path_dir = (args.save_path / args.vlm_type / sample_position)
                save_path_dir.mkdir(parents=True, exist_ok=True)
                clip_frames = torch.stack([sample_tensor(clip, sample_position) for clip in clips])
                with torch.no_grad():
                    preprocessed_frames = torch.Tensor(np.array(processor(clip_frames)['pixel_values'])).to(args.device)
                    clip_embeddings = model.get_image_features(preprocessed_frames)

                    clip_embeddings_numpy = clip_embeddings.detach().cpu().numpy()
                    save_path = save_path_dir / str(video_name)
                    np.save(save_path.with_suffix('.npy'), clip_embeddings_numpy)
        else:
            for sample_position in ['all', 'random_five']:
                save_path_dir = (args.save_path / args.vlm_type / sample_position)
                save_path_dir.mkdir(parents=True, exist_ok=True)
                clip_frames = [sample_tensor(clip, sample_position) for clip in clips]
                # breakpoint()
                with torch.no_grad():
                    # breakpoint()
                    clip_embeddings = []
                    # breakpoint()
                    for frames in clip_frames:
                        preprocessed_frames = torch.Tensor(np.array(processor(frames)['pixel_values'])).to(
                            args.device)
                        clip_embeddings_temp = model.get_image_features(preprocessed_frames)
                        clip_embeddings_numpy_temp = clip_embeddings_temp.detach().cpu().numpy()
                        clip_embeddings_numpy_temp = clip_embeddings_numpy_temp.mean(axis=0) # avg across all frames
                        clip_embeddings.append(clip_embeddings_numpy_temp)

                    clip_embeddings_numpy = np.stack(clip_embeddings)
                    # breakpoint()
                    save_path = save_path_dir / str(video_name)
                    np.save(save_path.with_suffix('.npy'), clip_embeddings_numpy)

    except Exception as e:
        print(f"Video {video_path} with error", e)
        continue

