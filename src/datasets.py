import json

import pandas as pd
from torch.utils.data import Dataset
import numpy as np
import os
import torch
import torch.nn.functional as F
from unidecode import unidecode
import pickle
class RGDataset(Dataset):
    def __init__(self, video_feat_path, label_path, clip_num=26, action_type='Ball',
                 score_type='Total_Score', train=True):
        if action_type == 'all' and not train:
            raise SystemError
        self.train = train
        self.video_path = video_feat_path
        self.erase_path = video_feat_path + '_erTrue'

        self.clip_num = clip_num
        self.labels = self.read_label(label_path, score_type, action_type)

    def read_label(self, label_path, score_type, action_type):
        fr = open(label_path, 'r')
        idx = {'Difficulty_Score': 1, 'Execution_Score': 2, 'Total_Score': 3}
        labels = []
        for i, line in enumerate(fr):
            if i == 0:
                continue
            line = line.strip().split()
            if action_type == 'all' or action_type == line[0].split('_')[0]:
                labels.append([line[0], float(line[idx[score_type]])])
        return labels

    def __getitem__(self, idx):
        video_feat = np.load(os.path.join(self.video_path, self.labels[idx][0] + '.npy'))

        # temporal random crop or padding
        if self.train:
            if len(video_feat) > self.clip_num:
                st = np.random.randint(0, len(video_feat) - self.clip_num)
                video_feat = video_feat[st:st + self.clip_num]
                # erase_feat = erase_feat[st:st + self.clip_num]
            elif len(video_feat) < self.clip_num:
                new_feat = np.zeros((self.clip_num, video_feat.shape[1]))
                new_feat[:video_feat.shape[0]] = video_feat
                video_feat = new_feat

        video_feat = torch.from_numpy(video_feat).float()
        return video_feat, self.normalize_score(self.labels[idx][1])

    def __len__(self):
        return len(self.labels)

    def normalize_score(self, score):
        return score / 25



class FisVDataset(Dataset):
    def __init__(self, video_feat_path, label_path, clip_num=26,
                 score_type='TES', train=True, vid_id_to_element_list_path='', action_mask_lookup="super_action_mask_lookup.pkl", action_masks=False, gdlt=False, vision_vlm_inject_path=None):
        self.train = train
        self.video_path = video_feat_path
        # self.erase_path = video_feat_path + '_erTrue'

        self.clip_num = clip_num
        self.labels = self.read_label(label_path, score_type)
        with open(vid_id_to_element_list_path) as f:
            self.element_lookup=json.load(f)
            self.element_lookup = {int(key): val for key, val in self.element_lookup.items()}
        # breakpoint()
        print("Length of Element Lookup", len(self.element_lookup))
        # print( self.element_lookup)
        self.score_type = score_type
        with open(action_mask_lookup, 'rb') as f:
            self.action_mask_lookup = pickle.load(f)
        self.action_masks = action_masks
        self.gdlt = gdlt
        self.vision_vlm_inject_path = vision_vlm_inject_path

    def action_label_to_supercategory(self, action_label):
        if 'Ch' in action_label:
            return 'choreographic_ss'
        if 'St' in action_label:
            return 'ss'
        if 'Sp' in action_label:
            return 'spin'
        return 'jump'

    def get_action_mask(self, supercategory):
        return self.action_mask_lookup["positive"][supercategory], self.action_mask_lookup["negative"][supercategory]

    def read_label(self, label_path, score_type):
        fr = open(label_path, 'r')
        idx = {'TES': 1, 'PCS': 2, 'Deductions': 3}
        labels = []
        for i, line in enumerate(fr):
            if i == 0:
                continue
            line = line.strip().split()
            labels.append([int(line[0]), float(line[idx[score_type]])])
        return labels

    def __getitem__(self, idx):
        video_feat = np.load(os.path.join(self.video_path, str(self.labels[idx][0]) + '.npy'))

        if self.vision_vlm_inject_path is not None:
            clip_inject_features = np.load(os.path.join(self.vision_vlm_inject_path, str(self.labels[idx][0]) + '.npy'))
            clip_inject_features = clip_inject_features[:video_feat.shape[0]]
            # print(video_feat.shape, clip_inject_features.shape)
        # temporal random crop or padding
        if self.train:
            if len(video_feat) > self.clip_num:
                st = np.random.randint(0, len(video_feat) - self.clip_num)
                video_feat = video_feat[st:st + self.clip_num]
                if self.vision_vlm_inject_path is not None:
                    clip_inject_features = clip_inject_features[st:st+self.clip_num]
                # erase_feat = erase_feat[st:st + self.clip_num]
            elif len(video_feat) < self.clip_num:
                new_feat = np.zeros((self.clip_num, video_feat.shape[1]))
                new_feat[:video_feat.shape[0]] = video_feat
                video_feat = new_feat
                if self.vision_vlm_inject_path is not None:
                    clip_inject_features_new = np.zeros((self.clip_num, clip_inject_features.shape[1]))
                    # print("--",clip_inject_features.shape, clip_inject_features_new.shape)
                    clip_inject_features_new[:clip_inject_features.shape[0]] = clip_inject_features
                    clip_inject_features = clip_inject_features_new
        # breakpoint()
        video_feat = torch.from_numpy(video_feat).float()
        if self.vision_vlm_inject_path is not None:
            clip_inject_features = torch.from_numpy(clip_inject_features).float()

        # breakpoint()
        if self.action_masks:
            pos_action_rubric_mask = torch.Tensor([self.get_action_mask(self.action_label_to_supercategory(bv_goe_data['element_name']))[0] for bv_goe_data in self.element_lookup[self.labels[idx][0]]])
            neg_action_rubric_mask = torch.Tensor([self.get_action_mask(self.action_label_to_supercategory(bv_goe_data['element_name']))[1] for bv_goe_data in self.element_lookup[self.labels[idx][0]]])
            if self.vision_vlm_inject_path is not None:
                return video_feat, clip_inject_features, torch.Tensor([bv_goe_data['goe'] for bv_goe_data in self.element_lookup[self.labels[idx][0]]]), torch.Tensor([bv_goe_data['base_value'] for bv_goe_data in self.element_lookup[self.labels[idx][0]]]), self.labels[idx][1], pos_action_rubric_mask, neg_action_rubric_mask #self.normalize_score(self.labels[idx][1])
            return video_feat, torch.Tensor([bv_goe_data['goe'] for bv_goe_data in self.element_lookup[self.labels[idx][0]]]), torch.Tensor([bv_goe_data['base_value'] for bv_goe_data in self.element_lookup[self.labels[idx][0]]]), self.labels[idx][1], pos_action_rubric_mask, neg_action_rubric_mask #self.normalize_score(self.labels[idx][1])
        if self.gdlt:
            if self.vision_vlm_inject_path is not None:
                return video_feat, clip_inject_features, torch.Tensor([self.normalize_score(bv_goe_data['goe']) for bv_goe_data in self.element_lookup[self.labels[idx][0]]]), torch.Tensor([self.normalize_score(bv_goe_data['base_value']) for bv_goe_data in self.element_lookup[self.labels[idx][0]]]), self.normalize_score(self.labels[idx][1])
            return video_feat, torch.Tensor([self.normalize_score(bv_goe_data['goe']) for bv_goe_data in self.element_lookup[self.labels[idx][0]]]), torch.Tensor([self.normalize_score(bv_goe_data['base_value']) for bv_goe_data in self.element_lookup[self.labels[idx][0]]]), self.normalize_score(self.labels[idx][1])
        if self.vision_vlm_inject_path is not None:
            return video_feat, clip_inject_features, torch.Tensor([bv_goe_data['goe'] for bv_goe_data in self.element_lookup[self.labels[idx][0]]]), torch.Tensor([bv_goe_data['base_value'] for bv_goe_data in self.element_lookup[self.labels[idx][0]]]), self.labels[idx][1] #self.normalize_score(self.labels[idx][1])
        return video_feat, torch.Tensor([bv_goe_data['goe'] for bv_goe_data in self.element_lookup[self.labels[idx][0]]]), torch.Tensor([bv_goe_data['base_value'] for bv_goe_data in self.element_lookup[self.labels[idx][0]]]), self.labels[idx][1] #self.normalize_score(self.labels[idx][1])

    def __len__(self):
        return len(self.labels)

    def normalize_score(self, score):
        if self.score_type == 'TES':
            norm_factor=42.81
        elif self.score_type == 'PCS':
            norm_factor=38.28
        else:
            norm_factor=80.75
        assert score <= norm_factor
        return score / norm_factor

class FisVPretrainingDataset(FisVDataset):
    def __init__(self, video_feat_path, label_path, clip_num=26,
                 score_type='TES', train=True, vid_id_to_element_list_path='', action_mask_lookup="super_action_mask_lookup.pkl", action_masks=False, gdlt=False, vision_vlm_inject_path=None, pretraining_threshold_pos=2, pretraining_threshold_neg=-2):
        super().__init__(video_feat_path, label_path, clip_num,
                 score_type, train, vid_id_to_element_list_path, action_mask_lookup, action_masks, gdlt, vision_vlm_inject_path)

        self.goe_thresh_positive = pretraining_threshold_pos
        self.goe_thresh_negative = pretraining_threshold_neg
        # filter a subset that is positive scoring and negative scoring
        self.indicies_labels, element_count = self.filter_extremes()
        print("Filtered by extreme:", len(self.indicies_labels), f"Num of Elements: {element_count}")
    def filter_extremes(self):
        routines_element_labels = []
        element_count = 0
        # breakpoint()
        for og_idx, label in enumerate(self.labels):

            # get the grade of execution by element
            goes = [bv_goe_data['goe'] for bv_goe_data in self.element_lookup[label[0]]]
            # filter by extremes
            elements_labels = [(element_idx, 1 if goe >= self.goe_thresh_positive else 0) for element_idx, goe in enumerate(goes) if goe >= self.goe_thresh_positive or goe <= self.goe_thresh_negative]
            if len(elements_labels) > 0:
                routines_element_labels.append((og_idx, elements_labels))
            element_count += len(elements_labels)
        return routines_element_labels, element_count

    def __len__(self):
        return len(self.indicies_labels)

    def __getitem__(self, idx):
        og_idx, label = self.indicies_labels[idx]
        old_train_val = self.train
        self.train = True # hack for validation loss
        video_feat = super().__getitem__(og_idx)[0] #np.load(os.path.join(self.video_path, str(self.labels[og_idx][0]) + '.npy'))
        self.train = old_train_val
        # action_label
        action_list = [self.action_label_to_supercategory(bv_goe_data['element_name']) for bv_goe_data in
             self.element_lookup[self.labels[og_idx][0]]]
        goes = [bv_goe_data['goe'] for bv_goe_data in self.element_lookup[self.labels[og_idx][0]]]
        difficulties = [bv_goe_data['base_value'] for bv_goe_data in self.element_lookup[self.labels[og_idx][0]]]
        element_names = [bv_goe_data['element_name'] for bv_goe_data in self.element_lookup[self.labels[og_idx][0]]]

        # print(action_list)
        # new_label_list = []
        element_indicies = []
        corresponding_actions = []
        corresponding_labels = []
        for i, _ in enumerate(action_list):
            if i < len(label):
                element_idx, pos_label = label[i]
                action_cls = action_list[element_idx]
            else:
                element_idx, pos_label = -1, -1 # padding
                action_cls=""
            element_indicies.append(element_idx)
            corresponding_actions.append(action_cls)
            corresponding_labels.append(pos_label)
        # print(corresponding_actions)
        element_info = list(zip(element_names, difficulties, goes, action_list))

        return video_feat, element_indicies, corresponding_actions, corresponding_labels, element_info # on model end, it will get either positive or negative

class FS800Dataset(Dataset):
    def __init__(self, video_feat_path, label_path, clip_num=26,
                 score_type='TES', train=True, vid_id_to_element_list_path='', action_mask_lookup="super_action_mask_lookup.pkl", action_masks=False, gdlt=False, vision_vlm_inject_path=None):
        self.train = train
        self.video_path = video_feat_path

        self.clip_num = clip_num
        self.indicies, self.labels = self.get_dataset(label_path) # only TES
        with open(vid_id_to_element_list_path) as f:
            self.element_lookup=json.load(f)
            self.element_lookup = {key: val for key, val in self.element_lookup.items()}

        print("Length of Element Lookup", len(self.element_lookup))

        self.score_type = score_type
        with open(action_mask_lookup, 'rb') as f:
            self.action_mask_lookup = pickle.load(f)
        self.action_masks = action_masks
        self.gdlt = gdlt
        # self.vision_vlm_inject_path = vision_vlm_inject_path

    def action_label_to_supercategory(self, action_label):
        if 'Ch' in action_label:
            return 'choreographic_ss'
        if 'St' in action_label:
            return 'ss'
        if 'Sp' in action_label:
            return 'spin'
        return 'jump'

    def get_action_mask(self, supercategory):
        return self.action_mask_lookup["positive"][supercategory], self.action_mask_lookup["negative"][supercategory]

    def get_dataset(self, label_path):
        def is_a_solo_program(name):
            for program_type in ['LS_', 'LF_', 'MS_', 'MF_']:
                if program_type in name:
                    return True
            return False
        unique_gender_length_map = {
            'LS': ('women', 'short'),
            'LF': ('women', 'long'),
            'MS': ('men', 'short'),
            'MF': ('men', 'long')
        }

        fr = open(label_path, 'r')
        labels = []
        ids = []
        for i, line in enumerate(fr):
            line = line.strip().split()
            if not is_a_solo_program(line[0]):
                continue
            ids.append(line[0])
            labels.append(float(line[1]))
        return ids, labels

    def __getitem__(self, idx):
        video_feat = np.load(os.path.join(self.video_path, self.indicies[idx] + '.npy')).mean(axis=1)
        # temporal random crop or padding
        if self.train:
            if len(video_feat) > self.clip_num:
                st = np.random.randint(0, len(video_feat) - self.clip_num)
                video_feat = video_feat[st:st + self.clip_num]
            elif len(video_feat) < self.clip_num:
                new_feat = np.zeros((self.clip_num, video_feat.shape[1]))
                new_feat[:video_feat.shape[0]] = video_feat
                video_feat = new_feat
                
        video_feat = torch.from_numpy(video_feat).float()
        set_limit = 12
        goes_tensor=torch.Tensor([bv_goe_data['goe'] for bv_goe_data in self.element_lookup[self.indicies[idx]]][:set_limit])
        bv_tensor=torch.Tensor([bv_goe_data['base_value'] for bv_goe_data in self.element_lookup[self.indicies[idx]]][:set_limit])
        padding_tensor=torch.Tensor([bv_goe_data['element_name'] != ''  for bv_goe_data in self.element_lookup[self.indicies[idx]]])

        if self.action_masks:
            pos_action_rubric_mask = torch.Tensor([self.get_action_mask(self.action_label_to_supercategory(bv_goe_data['element_name']))[0] for bv_goe_data in self.element_lookup[self.indicies[idx]]][:set_limit])
            neg_action_rubric_mask = torch.Tensor([self.get_action_mask(self.action_label_to_supercategory(bv_goe_data['element_name']))[1] for bv_goe_data in self.element_lookup[self.indicies[idx]]][:set_limit])
            return video_feat, goes_tensor, bv_tensor, self.labels[idx], pos_action_rubric_mask, neg_action_rubric_mask #self.normalize_score(self.labels[idx][1])
        if self.gdlt:
            return video_feat, torch.Tensor([self.normalize_score(bv_goe_data['goe']) for bv_goe_data in self.element_lookup[self.indicies[idx]]]), torch.Tensor([self.normalize_score(bv_goe_data['base_value']) for bv_goe_data in self.element_lookup[self.indicies[idx]]]), self.normalize_score(self.labels[idx])
        
        return video_feat, goes_tensor, bv_tensor, self.labels[idx] #self.normalize_score(self.labels[idx][1])

    def __len__(self):
        return len(self.labels)

    def normalize_score(self, score):
        if self.score_type == 'TES':
            norm_factor=121.24
        
        assert score <= norm_factor
        return score / norm_factor


class FS800PretrainingDataset(FS800Dataset):
    def __init__(self, video_feat_path, label_path, clip_num=26,
                 score_type='TES', train=True, vid_id_to_element_list_path='', action_mask_lookup="super_action_mask_lookup.pkl", action_masks=False, gdlt=False, vision_vlm_inject_path=None, pretraining_threshold_pos=2, pretraining_threshold_neg=-2):
        super().__init__(video_feat_path, label_path, clip_num,
                 score_type, train, vid_id_to_element_list_path, action_mask_lookup, action_masks, gdlt, vision_vlm_inject_path)

        self.goe_thresh_positive = pretraining_threshold_pos
        self.goe_thresh_negative = pretraining_threshold_neg
        # filter a subset that is positive scoring and negative scoring
        self.indicies_labels, element_count = self.filter_extremes()
        print("Filtered by extreme:", len(self.indicies_labels), f"Num of Elements: {element_count}")
    def filter_extremes(self):
        routines_element_labels = []
        element_count = 0
        # breakpoint()
        for og_idx, comp_skater_id in enumerate(self.indicies):

            # get the grade of execution by element
            goes = [bv_goe_data['goe'] for bv_goe_data in self.element_lookup[comp_skater_id]]
            # filter by extremes
            elements_labels = [(element_idx, 1 if goe >= self.goe_thresh_positive else 0) for element_idx, goe in enumerate(goes) if goe >= self.goe_thresh_positive or goe <= self.goe_thresh_negative]
            if len(elements_labels) > 0:
                routines_element_labels.append((og_idx, elements_labels))
            element_count += len(elements_labels)
        return routines_element_labels, element_count

    def __len__(self):
        return len(self.indicies_labels)

    def __getitem__(self, idx):
        og_idx, label = self.indicies_labels[idx]
        old_train_val = self.train
        self.train = True # hack for validation loss
        video_feat = super().__getitem__(og_idx)[0] #np.load(os.path.join(self.video_path, str(self.labels[og_idx][0]) + '.npy'))
        self.train = old_train_val
        # action_label
        action_list = [self.action_label_to_supercategory(bv_goe_data['element_name']) for bv_goe_data in
             self.element_lookup[self.indicies[og_idx]]]
        set_limit = 12

        goes = [bv_goe_data['goe'] for bv_goe_data in self.element_lookup[self.indicies[og_idx]]][:set_limit]
        difficulties = [bv_goe_data['base_value'] for bv_goe_data in self.element_lookup[self.indicies[og_idx]]][:set_limit]
        element_names = [bv_goe_data['element_name'] for bv_goe_data in self.element_lookup[self.indicies[og_idx]]][:set_limit]

        # print(action_list)
        # new_label_list = []
        element_indicies = []
        corresponding_actions = []
        corresponding_labels = []
        for i, _ in enumerate(action_list):
            if i < len(label):
                element_idx, pos_label = label[i]
                action_cls = action_list[element_idx]
            else:
                element_idx, pos_label = -1, -1 # padding
                action_cls=""
            element_indicies.append(element_idx)
            corresponding_actions.append(action_cls)
            corresponding_labels.append(pos_label)
        # print(corresponding_actions)
        element_info = list(zip(element_names, difficulties, goes, action_list))

        return video_feat, element_indicies, corresponding_actions, corresponding_labels, element_info # on model end, it will get either positive or negative
