import math
import os
import sys
import argparse
os.chdir('..')
sys.path.append(os.getcwd())

from datasets import RGDataset, FisVDataset, FisVPretrainingDataset
from torch.utils.data import DataLoader
from models import model, loss
import torch
import cv2
from tqdm import tqdm
import itertools

def get_fps(vid_id, video_dir):
    video_path = f'{video_dir}/{vid_id}.mp4'
    cap = cv2.VideoCapture(video_path)

    # Get the frames per second (fps) of the video
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Release the video capture object
    cap.release()
    if fps == 0:
        video_path = f'{video_dir}/{vid_id}.mov'
        cap = cv2.VideoCapture(video_path)

        # Get the frames per second (fps) of the video
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Release the video capture object
        cap.release()
    return fps

video_path='../data/fis-v/swintx_avg_fps25_clip32'
test_label_path='../data/fis-v/test.txt'
train_label_path='../data/fis-v/train.txt'
clip_num=128
score_type='TES'
# gdlt_fis_v_implicit_seg_128_clip_bv_kb_goe_pred_rescaling_1e-4_pred_bv_n_encoder_2_best
# gdlt_fis_v_implicit_seg_128_clip_bv_kb_goe_pred_rescaling_1e-4_pred_bv

def get_model(model_name, version='best', encoder_layers=1, device=0, predict_bv=False):
    from models import model, loss
    checkpoint = torch.load('./ckpt/' + f'{model_name}_{version}' + '.pkl')
    model = model.GDLT(1024, 256, 1, encoder_layers,
                       2, 7, 0.3, predict_base_values=predict_bv).to(device)
    model.load_state_dict(checkpoint)
    return model

def get_clip_gdlt_model(model_name, version='best', encoder_layers=1, decoder_layers=2, device=0,
                        vision_vlm_inject=False, restrict_first_quad=False):
    from models import model, loss
    checkpoint = torch.load(
        './ckpt/' + f'{model_name}_{version}' + '.pkl')
    model = model.CLIP_GDLT_W_Decoder(1024, 1, encoder_layers,
                                      decoder_layers, 7, 0.3,
                                      clip_embedding_path="./elements_preprocessing/text_embeddings_and_weights.pth",
                                      vision_vlm_inject=vision_vlm_inject, restrict_first_quad=restrict_first_quad).to(
        device)
    model.load_state_dict(checkpoint)
    model.clip_classifier.clip_classifier['positive_text_embeddings'] = model.clip_classifier.clip_classifier[
        'positive_text_embeddings'].to(device)
    model.clip_classifier.clip_classifier['negative_text_embeddings'] = model.clip_classifier.clip_classifier[
        'negative_text_embeddings'].to(device)

    model.clip_classifier.clip_classifier['pos_weights'] = model.clip_classifier.clip_classifier['pos_weights'].to(
        device)
    model.clip_classifier.clip_classifier['neg_weights'] = model.clip_classifier.clip_classifier['neg_weights'].to(
        device)

    return model

def build_clip_model(encoder_layers=1, decoder_layers=2, device=0, vision_vlm_inject=False, load_pretraining_ckpt=None, simplified=False):
    from models import model, loss
    model = model.CLIP_GDLT_W_Decoder(1024, 1, encoder_layers,
                                      decoder_layers, 7, 0.3, clip_embedding_path="./elements_preprocessing/text_embeddings_and_weights.pth", vision_vlm_inject=vision_vlm_inject, simplified=simplified).to(device)
    if load_pretraining_ckpt is not None:
        checkpoint = './ckpt/' + f'{load_pretraining_ckpt}' + '.pkl'
        state_dict = torch.load(checkpoint)
        model.load_state_dict(state_dict)
        print("Loaded weights at:", checkpoint)
    model.clip_classifier.clip_classifier['positive_text_embeddings']=model.clip_classifier.clip_classifier['positive_text_embeddings'].to(device)
    model.clip_classifier.clip_classifier['negative_text_embeddings']=model.clip_classifier.clip_classifier['negative_text_embeddings'].to(device)
    # model.clip_classifier.clip_classifier=clip_classifier_pooler
    pos_zeros_rubric = torch.zeros(model.clip_classifier.clip_classifier['positive_text_embeddings'].shape).to(device)
    neg_zeros_rubric = torch.zeros(model.clip_classifier.clip_classifier['negative_text_embeddings'].shape).to(device)

    model.clip_classifier.clip_classifier['max_positive_text_embeddings']=torch.maximum(model.clip_classifier.clip_classifier['positive_text_embeddings'], pos_zeros_rubric).to(device)
    model.clip_classifier.clip_classifier['max_negative_text_embeddings']=torch.maximum(model.clip_classifier.clip_classifier['negative_text_embeddings'], neg_zeros_rubric).to(device)

    # print(model.clip_classifier.clip_classifier['max_positive_text_embeddings'])
    return model

train_data = FisVDataset(video_path, train_label_path, clip_num=clip_num,
                          score_type=score_type, train=True)
test_data = FisVDataset(video_path, test_label_path, clip_num=clip_num,
                          score_type=score_type, train=False)
test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0)
train_loader = DataLoader(train_data, batch_size=1, shuffle=False, num_workers=0)


def get_attention_peaks(attns):
    highest_attention_indices = torch.argmax(attns[-1], axis=1)
    return highest_attention_indices


def convert_from_seconds_to_clip_num(seconds, fps, if_end=False, frames_per_clip=32):
    if if_end:
        return math.ceil((seconds + if_end) * fps / frames_per_clip)
    else:
        return math.floor((seconds) * fps / frames_per_clip)


def within_range(x, start, end):
    if (x < end) and (x > start):
        return True
    else:
        return False


def get_segmentation_mask(start, end, number_of_samples):
    segmentation_mask = torch.zeros(number_of_samples)
    segmentation_mask[start:end] = 1
    return segmentation_mask


def get_segmentation_mask_from_attn_mask(attn_mask, threshold=0.001):
    segmentation_mask = torch.zeros(attn_mask.shape[-1])
    segmentation_mask = torch.where(attn_mask > threshold, torch.tensor(1), torch.tensor(0))

    return segmentation_mask

import torch
def find_idx(video_id, dataset):
    for idx, item in enumerate(dataset.labels):
        if video_id == dataset.labels[0]:
            return idx
    return -1

def get_attention_mask(video_id, model, device):
    video_feat, gt_goes, base_values, label = test_loader.dataset[find_idx(video_id, test_loader.dataset)]
    model.eval()
    with torch.no_grad():
        video_feat = video_feat.to(device).unsqueeze(0)
        base_values = base_values.to(device).unsqueeze(0)
        # breakpoint()
        output_dict = model(video_feat, base_values, rescaling=True, need_weights=True)
        attns = output_dict['attns'][-1]

    return attns.cpu()

def calculate_iou(mask1, mask2):
    # Convert masks to boolean tensors
    mask1_bool = mask1.bool()
    mask2_bool = mask2.bool()

    # Calculate intersection and union
    intersection = torch.logical_and(mask1_bool, mask2_bool).float().sum()
    union = torch.logical_or(mask1_bool, mask2_bool).float().sum()

    # Avoid division by zero
    epsilon = 1e-5
    iou = intersection / (union + epsilon)

    return iou.item()

def IO_accuracy(predicted_peaks, ground_truth):
    hits = 0
    total = 0

    for vid_idx in range(len(predicted_peaks)):
        for element_idx in range(len(predicted_peaks[vid_idx])):
            peak = predicted_peaks[vid_idx][element_idx]
            annotated_segment = ground_truth[vid_idx][element_idx]
            if within_range(peak, annotated_segment[0], annotated_segment[1]): # 0:start_clip and 1:end_clip
                hits+=1
            total +=1
            # print(peak, annotated_segment)
    return hits/total

def OO_accuracy(predicted_peaks, ground_truth):
    hits = 0
    total = 0

    for vid_idx in range(len(predicted_peaks)):
        for element_idx in range(len(predicted_peaks[vid_idx])):
            peak = predicted_peaks[vid_idx][element_idx]
            for annotated_segment in ground_truth[vid_idx]:
                if within_range(peak, annotated_segment[0], annotated_segment[1]): # 0:start_clip and 1:end_clip
                    hits+=1
                    break
            total+=1
    return hits/total

def OO_1_1_accuracy(predicted_peaks, ground_truth):
    hits = 0
    total = 0

    def calc_hits_per_video(peaks_per_query, gt_videos, element_num=7):
        local_hits = 0
        for element_idx in range(element_num):
            peak = peaks_per_query[element_idx]
            annotated_segment = gt_videos[element_idx]
            if within_range(peak, annotated_segment[0], annotated_segment[1]): # 0:start_clip and 1:end_clip
                local_hits+=1
        return local_hits

    for vid_idx in range(len(predicted_peaks)):
        gt_permutations = list(itertools.permutations(ground_truth[vid_idx], 7))
        # print(len(gt_permutations))
        # exit()
        hits += max([calc_hits_per_video(predicted_peaks[vid_idx], gt_permutation) for gt_permutation in gt_permutations])
        total += 7
        
    return hits/total
# def OO_1_1_accuracy(predicted_peaks, ground_truth):
#     hits = 0
#     total = 0
#
#     for vid_idx in range(len(predicted_peaks)):
#         fps = fps_list[vid_idx]
#         for element_idx in range(len(predicted_peaks[vid_idx])):
#             # something
#     return hits/total

def evaluate(model, df, args):
    def datetime_str_to_seconds(datetime_str):
        minutes, seconds = map(int, datetime_str.split(':'))
        return minutes * 60 + seconds

    model.eval()
    predicted_peaks = []
    ground_truth_segments = []
    for _, row in df.iterrows():
        fps = row['fps']
        attn_mask = get_attention_mask(row["video_id"], model, device=args.device)
        attention_peaks_indicies = get_attention_peaks(attn_mask)
        predicted_peaks_vid= []
        gt_annotated_vid = []
        for element_idx, peak in enumerate(attention_peaks_indicies):
            start_clip = convert_from_seconds_to_clip_num(
                datetime_str_to_seconds(row[f"element_{element_idx + 1}_start"]), fps=fps)
            end_clip = convert_from_seconds_to_clip_num(datetime_str_to_seconds(row[f"element_{element_idx + 1}_end"]),
                                                        fps=fps, if_end=True)
            predicted_peaks_vid.append(attention_peaks_indicies[element_idx])
            gt_annotated_vid.append((start_clip, end_clip))
        predicted_peaks.append(predicted_peaks_vid)
        ground_truth_segments.append(gt_annotated_vid)
    print("OO 1:1 Acc: ", round(OO_1_1_accuracy(predicted_peaks, ground_truth_segments), 3)*100)
    print("IO Acc: ", round(IO_accuracy(predicted_peaks, ground_truth_segments), 3)*100)
    print("OO Acc: ", round(OO_accuracy(predicted_peaks, ground_truth_segments), 3)*100)

def main():
    # python analysis/implicit_segmentation_eval.py --model_name MODEL_NAME --simplified_rubric --device 0
    parser = argparse.ArgumentParser(description="Process some parameters.")

    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to use.")
    parser.add_argument("--simplified_rubric", action='store_true', help="Enable simplified rubric")
    parser.add_argument("--device", type=int, required=True, help="Device number to use for computation")

    args = parser.parse_args()

    print(f"Model Name: {args.model_name}")

    ckpt_name = args.model_name + '_best'
    model = build_clip_model(encoder_layers=1, decoder_layers=2, device=args.device, vision_vlm_inject=False,
                             load_pretraining_ckpt=ckpt_name, simplified=args.simplified_rubric)
    import pandas as pd
    df = pd.read_csv('../data/segment_annotations.csv') 
    df['fps'] = df['video_id'].apply(lambda vid_id: get_fps(vid_id))

    evaluate(model, df, args)

if __name__ == "__main__":
    main()