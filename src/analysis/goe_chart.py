import argparse
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import os
import sys
from tqdm import tqdm
from collections import defaultdict
plt.rcParams.update({'font.size': 24})  # Change 14 to your desired font size

print(os.getcwd())
sys.path.append(os.getcwd())

from datasets import RGDataset, FisVDataset
from torch.utils.data import DataLoader
from models import model, loss
import torch
import torch.nn.functional as F

video_path = '../data/fis-v/swintx_avg_fps25_clip32'
test_label_path = '../data/fis-v/test.txt'
train_label_path = '../data/fis-v/train.txt'
clip_num = 128
score_type = 'TES'

def get_model(model_name, version='best', encoder_layers=1, predict_bv=False):
    from models import model, loss
    checkpoint = torch.load(
        './ckpt/' + f'{model_name}_{version}' + '.pkl')
    device = 2
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
                                      clip_embedding_path=" ./elements_preprocessing/text_embeddings_and_weights.pth",
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


train_data = FisVDataset(video_path, train_label_path, clip_num=clip_num,
                         score_type=score_type, train=True, action_masks=True)
test_data = FisVDataset(video_path, test_label_path, clip_num=clip_num,
                        score_type=score_type, train=False, action_masks=True)
test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0)
train_loader = DataLoader(train_data, batch_size=1, shuffle=False, num_workers=0)


# test out how close the clip embeddings are compared to outputs when newly initialized
def get_initial(clip_model, x, base_model, pos_action_mask=None, neg_action_mask=None):
    old_shape = x.shape
    # x = torch.maximum(x, torch.zeros(x.shape).to(x.device))

    tensor1_reshaped = x.view(-1, old_shape[-1])
    positive_rubric_cos_sim = F.cosine_similarity(tensor1_reshaped[:, None, :], clip_model.clip_classifier['positive_text_embeddings'][None, :, :], dim=2)
    negative_rubric_cos_sim = F.cosine_similarity(tensor1_reshaped[:, None, :], clip_model.clip_classifier['negative_text_embeddings'][None, :, :], dim=2)
    return tensor1_reshaped, positive_rubric_cos_sim, negative_rubric_cos_sim

    # return tensor1_reshaped, positive_rubric_cos_sim.max(), positive_rubric_cos_sim.min(), negative_rubric_cos_sim.max(), negative_rubric_cos_sim.min()

def model_fwd_initial(model, x, base_values, pos_action_mask=None, neg_action_mask=None, need_weights=False, vision_vlm_inject=None):
    b, t, c = x.shape
    x = model.in_proj(x.transpose(1, 2)).transpose(1, 2)

    q = model.segment_anchors.weight.unsqueeze(0).repeat(b, 1, 1)
    encode_x = model.transformer.encoder(x)
    if vision_vlm_inject is not None:
        encode_x+=clip_vision_inject
    attns=[]
    if need_weights:
        q1, attns = model.transformer.decoder(q, encode_x, need_weights=need_weights)
        return get_initial(model.clip_classifier, q1, base_values, pos_action_mask, neg_action_mask), attns
    else:
        q1 = model.transformer.decoder(q, encode_x)
        return get_initial(model.clip_classifier, q1, base_values, pos_action_mask, neg_action_mask)


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

import numpy as np

def assign_to_bin(value):
    bins = np.array([-3, -2, -1, 0, 1, 2, 3])  # +0.5

    # Using numpy.digitize to find the appropriate bin for the value
    bin_index = np.digitize(value, bins, right=True)

    return bins[bin_index]


# iterate through dataset, extract per decoder rubric activations, positive and negative, and the goe corresponding to that element

def get_cossims_by_gt_goe(model, data_loader, device):
    model.eval()
    x_decoder_outs = []
    pos_sim_maxs = []
    pos_sim_mins = []
    neg_sim_maxs = []
    neg_sim_mins = []
    goe_bin_cossim_lookup_positive = defaultdict(list)
    goe_bin_cossim_lookup_negative = defaultdict(list)

    for video_feat, gt_goes, base_values, label, pos_action_rubric_mask, neg_action_rubric_mask in tqdm(data_loader):
        with torch.no_grad():
            video_feat = video_feat.to(device)
            base_values = base_values.to(device)
            label = label.float().to(device)
            if pos_action_rubric_mask is not None:
                pos_action_rubric_mask = pos_action_rubric_mask.to(device)
                neg_action_rubric_mask = neg_action_rubric_mask.to(device)

            x = video_feat
            (x_decoder_out, pos_sim, neg_sim), attns = model_fwd_initial(model, x, base_values, pos_action_rubric_mask,
                                                                         neg_action_rubric_mask, need_weights=True)
            # for la
            for element_idx, gt_goe in enumerate(gt_goes):
                goe_key = assign_to_bin(gt_goe)[0]
                goe_bin_cossim_lookup_positive[goe_key].extend(pos_sim[element_idx].tolist())
                goe_bin_cossim_lookup_negative[goe_key].extend(neg_sim[element_idx].tolist())

            x_decoder_outs.append(x_decoder_out.detach())
            pos_sim_maxs.append(pos_sim.max().item())
            pos_sim_mins.append(pos_sim.min().item())
            neg_sim_maxs.append(neg_sim.max().item())
            neg_sim_mins.append(neg_sim.min().item())
    return goe_bin_cossim_lookup_positive, goe_bin_cossim_lookup_negative

def get_graphic(args, goe_bin_cossim_lookup_positive, goe_bin_cossim_lookup_negative):
    data = []
    for goe_key in goe_bin_cossim_lookup_positive:
        data.append({
            "GT Grade of Execution Bins": goe_key,
            "Type": "Positive",
            "Cosine Similarity": goe_bin_cossim_lookup_positive[goe_key]
        })
        data.append({
            "GT Grade of Execution Bins": goe_key,
            "Type": " Negative",
            "Cosine Similarity": goe_bin_cossim_lookup_negative[goe_key]
        })

    df = pd.DataFrame(data)
    # We need to explode the Distance lists into separate rows
    df = df.explode('Cosine Similarity')
    df['Cosine Similarity'] = df['Cosine Similarity'].astype(float)  # Ensure Distance is float

    # Plotting
    fig, ax = plt.subplots(figsize=(20, 8))
    sns.boxplot(x='GT Grade of Execution Bins', y='Cosine Similarity', hue='Type', data=df, palette="Set2")
    plt.xticks(rotation=45)
    plt.title(f'Element-Rubric Text Alignment vs GOE ({args.title})')
    plt.tight_layout()
    plt.savefig(f'./outputs/{args.save_name}.png', format='png', dpi=300)



def main():
    # python analysis/goe_chart.py --model_name MODEL_NAME --title "TITLE" --save_name "SAVE_NAME" --simplified_rubric --mode train --device 0
    parser = argparse.ArgumentParser(description="Process some parameters.")

    # Argument for the model name
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to use.")

    # Argument for the title
    parser.add_argument("--title", type=str, required=True, help="Title of the operation.")

    # Argument for the save path
    parser.add_argument("--save_name", type=str, required=True, help="Path where the results will be saved.")
    # Flag for simplified rubric
    parser.add_argument("--simplified_rubric", action='store_true', help="Enable simplified rubric")
    parser.add_argument("--FT", action='store_true', help="finetuning")

    # Argument for specifying mode as either 'train' or 'test'
    parser.add_argument("--mode", type=str, choices=['train', 'test'], required=True,
                        help="Specify the mode: train or test")
    parser.add_argument("--device", type=int, required=True, help="Device number to use for computation")

    args = parser.parse_args()

    # Print arguments to verify (can be replaced with actual function calls)
    print(f"Model Name: {args.model_name}")
    print(f"Title: {args.title}")
    print(f"Save Name: {args.save_name}")
    ckpt_name = args.model_name + ('_best' if args.FT else '_pretraining_best')

    model = build_clip_model(encoder_layers=1, decoder_layers=2, device=args.device, vision_vlm_inject=False,
                             load_pretraining_ckpt=ckpt_name, simplified=args.simplified_rubric)

    dataset = train_data if args.mode == 'train' else test_data
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    goe_bin_cossim_lookup_positive, goe_bin_cossim_lookup_negative = get_cossims_by_gt_goe(model, data_loader=data_loader, device=args.device)
    get_graphic(args, goe_bin_cossim_lookup_positive, goe_bin_cossim_lookup_negative)



if __name__ == "__main__":
    main()
