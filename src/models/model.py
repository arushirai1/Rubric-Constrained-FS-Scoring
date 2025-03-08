import pickle

from torch import nn
import torch
import torch.nn.functional as F
import numpy as np
import copy
from models.transformer import Transformer
from pathlib import Path

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        try:
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0)
        except AttributeError:
            pass


# 主力model
class GDLT(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_head, n_encoder, n_decoder, n_query, dropout, predict_base_values=False, deduction_modeling=False, gdlt=False):
        super(GDLT, self).__init__()
        self.in_proj = nn.Sequential(
            nn.Conv1d(kernel_size=1, in_channels=in_dim, out_channels=in_dim // 2),
            nn.BatchNorm1d(in_dim // 2),
            nn.ReLU(),
            nn.Conv1d(kernel_size=1, in_channels=in_dim // 2, out_channels=hidden_dim),
            nn.BatchNorm1d(hidden_dim)
        )

        self.transformer = Transformer(
            d_model=hidden_dim,
            nhead=n_head,
            num_encoder_layers=n_encoder,
            num_decoder_layers=n_decoder,
            dim_feedforward=3 * hidden_dim,
            batch_first=True,
            dropout=dropout
        )

        self.segment_anchors = nn.Embedding(n_query, hidden_dim)

        self.weight = torch.linspace(0, 1, n_query, requires_grad=False).cuda()
        tunable_weights=True if not gdlt else False
        if tunable_weights:
            self.weight = torch.linspace(0, 1, n_query, requires_grad=True).cuda()
            #nn.init.normal_(self.weight , std=0.02)
            self.weight = nn.Parameter(self.weight)
        print(self.weight)
        if not gdlt:
            self.regressor = nn.Linear(hidden_dim, 1)#n_query)
        else:
            self.regressor = nn.Linear(hidden_dim, n_query)

        self.predict_base_values=predict_base_values
        if self.predict_base_values:
            self.base_value_regressor = nn.Linear(hidden_dim, 1)
        self.deduction_modeling = deduction_modeling
        if self.deduction_modeling:
            self.deduction_regressor = nn.Linear(hidden_dim, 1)

    def forward(self, x, base_values, rescaling=True, need_weights=False, gdlt=False, clip_vision_inject=None):
        # x (b, t, c)
        b, t, c = x.shape
        x = self.in_proj(x.transpose(1, 2)).transpose(1, 2)

        q = self.segment_anchors.weight.unsqueeze(0).repeat(b, 1, 1)
        encode_x = self.transformer.encoder(x)
        if clip_vision_inject is not None:
            encode_x+=clip_vision_inject
        attns=[]
        if need_weights:
            q1, attns = self.transformer.decoder(q, encode_x, need_weights=need_weights)

        else:
            q1 = self.transformer.decoder(q, encode_x)
        # print(q1.shape)
        s = self.regressor(q1)  # (b, n, n)
        base_values_pred = base_values

        if gdlt:
            s = torch.diagonal(s, dim1=-2, dim2=-1)  # (b, n)
            norm_s = torch.sigmoid(s)
            norm_s = norm_s / torch.sum(norm_s, dim=1, keepdim=True)
            if len(base_values_pred.shape) == 1:
                out = torch.sum(self.weight.unsqueeze(0).repeat(b, 1) * norm_s, dim=1)
            else:
                out = torch.sum(self.weight.unsqueeze(0).repeat(b, 1) * norm_s, dim=1) + torch.sum(base_values_pred, dim=1)
            goe_prediction=0
        else:
            if self.predict_base_values:
                base_values_pred = self.base_value_regressor(q1).squeeze(2)

            # print(s.shape)
            # breakpoint()
            norm_s = torch.sigmoid(s)

            if rescaling:
                goe_prediction = (norm_s.squeeze(2) * 10) - 5
            else:
                goe_prediction = s.squeeze(2)
            out = goe_prediction + base_values_pred  # no rescaling
            out = torch.sum(out, dim=1)

        # out = torch.sigmoid(torch.sum(out, dim=1))
        # s = torch.diagonal(s, dim1=-2, dim2=-1)  # (b, n)
        # norm_s = torch.sigmoid(s)
        # norm_s = norm_s / torch.sum(norm_s, dim=1, keepdim=True)
        # out = torch.sum(self.weight.unsqueeze(0).repeat(b, 1) * norm_s, dim=1)

        return {'output': out, 'embed': q1, 'attns': attns, 'base_value_predictions': base_values_pred, "goe_predictions": goe_prediction}

class CLIP_GDLT_W_Decoder(nn.Module):
    def __init__(self, in_dim, n_head, n_encoder, n_decoder, n_query, dropout, clip_embedding_path, l1_norm=False, use_sigmoid=False, use_relu=False, vision_vlm_inject=False, restrict_first_quad=False, simplified=False, is_rg=False):
        super(CLIP_GDLT_W_Decoder, self).__init__()
        hidden_dim = 768 # hidden dim should be the same size as clip

        self.in_proj = nn.Sequential(
            nn.Conv1d(kernel_size=1, in_channels=in_dim, out_channels=in_dim // 2),
            nn.BatchNorm1d(in_dim // 2),
            nn.ReLU(),
            nn.Conv1d(kernel_size=1, in_channels=in_dim // 2, out_channels=hidden_dim),
            nn.BatchNorm1d(hidden_dim)
        )

        self.transformer = Transformer(
            d_model=hidden_dim,
            nhead=n_head,
            num_encoder_layers=n_encoder,
            num_decoder_layers=n_decoder,
            dim_feedforward=1 * hidden_dim,
            batch_first=True,
            dropout=dropout
        )

        self.segment_anchors = nn.Embedding(n_query, hidden_dim)
        self.clip_classifier=CLIPClassifer(clip_embedding_path, l1_norm, use_sigmoid, use_relu, restrict_first_quad, simplified=simplified, is_rg=is_rg)
        self.vision_vlm_inject=vision_vlm_inject

    def pretraining_forward(self, x, need_weights=False, clip_vision_inject=None):
        # x (b, t, c)
        b, t, c = x.shape
        x = self.in_proj(x.transpose(1, 2)).transpose(1, 2)

        q = self.segment_anchors.weight.unsqueeze(0).repeat(b, 1, 1)
        encode_x = self.transformer.encoder(x)
        if self.vision_vlm_inject:
            encode_x += clip_vision_inject # concat
        attns=[]
        if need_weights:
            q1, attns = self.transformer.decoder(q, encode_x, need_weights=need_weights)
        else:
            q1 = self.transformer.decoder(q, encode_x)
        return q1, attns

    def forward(self, x, base_values, pos_action_mask=None, neg_action_mask=None, rescaling=True, need_weights=False, clip_vision_inject=None):
        # x (b, t, c)
        b, t, c = x.shape
        x = self.in_proj(x.transpose(1, 2)).transpose(1, 2)

        q = self.segment_anchors.weight.unsqueeze(0).repeat(b, 1, 1)
        encode_x = self.transformer.encoder(x)
        if self.vision_vlm_inject:
            encode_x += clip_vision_inject # concat
        attns=[]
        if need_weights:
            q1, attns = self.transformer.decoder(q, encode_x, need_weights=need_weights)
        else:
            q1 = self.transformer.decoder(q, encode_x)

        out = self.clip_classifier(q1, base_values, pos_action_mask, neg_action_mask, rescaling)
        return {**out, 'embed': q1, 'attns': attns}

class CLIP_GDLT(nn.Module):
    def __init__(self, in_dim, n_head, n_encoder, dropout, clip_embedding_path):
        super(CLIP_GDLT, self).__init__()
        hidden_dim = 768 # hidden dim should be the same size as clip
        self.in_proj = nn.Sequential(
            nn.Conv1d(kernel_size=1, in_channels=in_dim, out_channels=in_dim // 2),
            nn.BatchNorm1d(in_dim // 2),
            nn.ReLU(),
            nn.Conv1d(kernel_size=1, in_channels=in_dim // 2, out_channels=hidden_dim),
            nn.BatchNorm1d(hidden_dim)
        )

        self.transformer = Transformer(
            d_model=hidden_dim,
            nhead=n_head,
            num_encoder_layers=n_encoder,
            num_decoder_layers=0,
            dim_feedforward=3 * hidden_dim,
            batch_first=True,
            dropout=dropout
        )
        self.clip_classifier=CLIPClassifer(clip_embedding_path)

    def forward(self, x, base_values, rescaling=True, need_weights=False):
        # b, t, c = x.shape
        x = self.in_proj(x.transpose(1, 2)).transpose(1, 2)
        encode_x = self.transformer.encoder(x)
        out = self.clip_classifier(encode_x,base_values)
        return out

class CLIPClassifer(nn.Module):
    def __init__(self, path_to_clip_classifier, l1_norm, use_sigmoid, use_relu, restrict_first_quad, simplified, is_rg=False):
        super(CLIPClassifer, self).__init__()
        if not Path(path_to_clip_classifier).exists():
            print("invalid clip classifier path")
            exit(0)
        self.restrict_first_quad = restrict_first_quad
        self.is_rg = is_rg
        
        if simplified:
            from models.text_encoder import TextEncoder
            from data.rubric_items import RubricItems

            text_encoder = TextEncoder("clip", aggregation_method="embed")
            # self.text_prompt = f"a photo of a %s"
            # get text embeddings for rubric items
            rubric_items = RubricItems(simplified, text_prompt=False, is_rg=self.is_rg)
            self.clip_classifier = torch.load(path_to_clip_classifier)

            if self.is_rg:
                positives, pos_weights = rubric_items.get_positives()
                negatives, neg_weights = rubric_items.get_negatives()

                self.clip_classifier['neg_weights'] = neg_weights
                self.clip_classifier['pos_weights'] = pos_weights

            else:
                positives = rubric_items.get_positives()
                negatives = rubric_items.get_negatives()
            get_rubric_embeddings = lambda rubric_list: torch.stack([text_encoder.get_embeddings(rubric_item) for rubric_item in rubric_list])

            self.clip_classifier['positive_text_embeddings'] = get_rubric_embeddings(positives)
            self.clip_classifier['negative_text_embeddings'] = get_rubric_embeddings(negatives)
        else:
            self.clip_classifier = torch.load(path_to_clip_classifier)

        for key in self.clip_classifier:
            if isinstance(self.clip_classifier[key], list):
                self.clip_classifier[key] = torch.Tensor(self.clip_classifier[key])
            self.clip_classifier[key]=self.clip_classifier[key].cuda()
        if self.restrict_first_quad:
            pos_zeros_rubric = torch.zeros(self.clip_classifier['positive_text_embeddings'].shape).cuda()
            neg_zeros_rubric = torch.zeros(self.clip_classifier['negative_text_embeddings'].shape).cuda()
            self.clip_classifier['max_positive_text_embeddings'] = torch.maximum(
                self.clip_classifier['positive_text_embeddings'], pos_zeros_rubric).cuda()
            self.clip_classifier['max_negative_text_embeddings'] = torch.maximum(
                self.clip_classifier['negative_text_embeddings'], neg_zeros_rubric).cuda()

        self.l1_norm = l1_norm
        self.use_sigmoid=use_sigmoid
        self.scaling_factor=4
        self.use_relu=use_relu
    def forward(self, x, base_values, pos_action_mask=None, neg_action_mask=None, rescaling=False):
        # x (b, t, c)
        # clip_text_video_similarities =torch.matmul(x, self.clip_classifier['positive_text_embeddings'].t())
        # neg_clip_text_video_similarities =torch.matmul(x, self.clip_classifier['negative_text_embeddings'].t())
        old_shape = x.shape
        key_prefix=""
        if self.restrict_first_quad:
            x = torch.maximum(x, torch.zeros(x.shape).cuda())
            key_prefix="max_"
        tensor1_reshaped = x.view(-1, old_shape[-1])
        clip_text_video_activations = F.cosine_similarity(tensor1_reshaped[:, None, :], self.clip_classifier[key_prefix+'positive_text_embeddings'][None, :, :], dim=2)
        neg_clip_text_video_activations = F.cosine_similarity(tensor1_reshaped[:, None, :], self.clip_classifier[key_prefix+'negative_text_embeddings'][None, :, :], dim=2)
        clip_text_video_activations = clip_text_video_activations.view(old_shape[0], old_shape[1], -1 )
        neg_clip_text_video_activations = neg_clip_text_video_activations.view(old_shape[0], old_shape[1], -1)

        # clip_text_video_activations = torch.softmax(clip_text_video_similarities, 2)
        # neg_clip_text_video_activations = torch.softmax(neg_clip_text_video_similarities, 2)
        #
        if self.use_sigmoid:
            clip_text_video_activations = torch.sigmoid(clip_text_video_activations*self.scaling_factor)
            neg_clip_text_video_activations = torch.sigmoid(neg_clip_text_video_activations*self.scaling_factor)
        elif self.use_relu:
            clip_text_video_activations = torch.relu(clip_text_video_activations)
            neg_clip_text_video_activations = torch.relu(neg_clip_text_video_activations)
        if pos_action_mask is not None and neg_action_mask is not None:
            clip_text_video_activations = clip_text_video_activations*pos_action_mask
            neg_clip_text_video_activations = neg_clip_text_video_activations*neg_action_mask
        goe_predictions = torch.sum(clip_text_video_activations * self.clip_classifier['pos_weights'], dim=2)+torch.sum(neg_clip_text_video_activations * self.clip_classifier['neg_weights'], dim=2)

        if self.is_rg:
            out = 10 - torch.sum(goe_predictions, dim=1)
            if rescaling:
                out = torch.sigmoid(out) * 10 # keep score between 0-10
            print(out)
        else:
            if rescaling:
                # breakpoint()
                if goe_predictions.shape[-1] == 7:
                    goe_predictions = -3 + (torch.sigmoid(goe_predictions) * 6)  # linear transformation
                else:
                    goe_predictions = -5 + (torch.sigmoid(goe_predictions) * 10)  # linear transformation
            out = torch.sum(goe_predictions, dim=1) + torch.sum(base_values, dim=1)

        return_dict = {'output': out, 'goe_predictions': goe_predictions, 'pos_rubric_cos_sims': clip_text_video_activations, 'neg_rubric_cos_sims':neg_clip_text_video_activations, 'losses':{}}

        if self.l1_norm:
            l1_norm_loss = torch.norm(clip_text_video_activations, p=1) + torch.norm(neg_clip_text_video_activations, p=1)
            return_dict['losses']['l1_norm'] = 0.05 * l1_norm_loss # scaling factor

        return return_dict