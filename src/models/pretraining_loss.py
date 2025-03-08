import random

from models.text_encoder import TextEncoder
import torch
from torch import nn
import torch.nn.functional as F
from collections import defaultdict
from data.rubric_items import RubricItems

class PretrainingLossFn(nn.Module):
    def __init__(self, text_model_type, aggregation_method,  margin=0.5, enforce_dissimilarity=False, randomly_sample_dissim=False, metric='cosine', simplified=False, use_text_prompt=True, hand_negatives=False, visual_text_triplet_loss=False, num_queries=12):
        super(PretrainingLossFn, self).__init__()
        self.text_encoder = TextEncoder(text_model_type, aggregation_method= aggregation_method)
        self.text_prompt = f"a photo of a %s"
        # get text embeddings for action labels
        self.action_embedding_lookup = [self.text_encoder.get_embeddings(self.text_prompt.format(action)) for action in ['jump', 'step sequence', 'choreographic step sequence', 'spin']]
        self.hand_negatives = hand_negatives
        # get text embeddings for rubric items
        rubric_items = RubricItems(simplified, use_text_prompt)
        positives = rubric_items.get_positives()
        negatives = rubric_items.get_negatives()
        self.positive_rubric_items = self.restructre_criteria(positives)
        self.negative_rubric_items = self.restructre_criteria(negatives)
        self.positive_rubric_embeddings = {action: torch.stack([self.text_encoder.get_embeddings(rubric_item) for rubric_item in rubric_list]).cuda() for action, rubric_list in self.positive_rubric_items.items()}
        self.negative_rubric_embeddings = {action: torch.stack([self.text_encoder.get_embeddings(rubric_item) for rubric_item in rubric_list]).cuda() for action, rubric_list in self.negative_rubric_items.items()}
        if self.hand_negatives:
            # breakpoint()
            if not simplified:
                raise Exception("Must use simplified rubric text with hand negatives")
            hand_negatives_positive_rubric_items = self.restructre_criteria(rubric_items.get_hand_negatives(positive_rubric=True))
            hand_negatives_negative_rubric_items = self.restructre_criteria(rubric_items.get_hand_negatives(positive_rubric=False))
            self.hand_negatives_for_positive_rubric_embeddings = {action: torch.stack([self.text_encoder.get_embeddings(rubric_item) for rubric_item in rubric_list]).cuda() for action, rubric_list in hand_negatives_positive_rubric_items.items()}
            self.hand_negatives_for_negative_rubric_embeddings = {action: torch.stack([self.text_encoder.get_embeddings(rubric_item) for rubric_item in rubric_list]).cuda() for action, rubric_list in hand_negatives_negative_rubric_items.items()}

        self.loss_fn = torch.nn.CosineEmbeddingLoss(margin=margin, reduction='none')
        self.enforce_dissimilarity = enforce_dissimilarity
        self.randomly_sample_dissim = randomly_sample_dissim
        self.metric = metric
        self.margin = margin
        self.use_visual_text_triplet_loss = visual_text_triplet_loss
        self.num_queries = num_queries
    def restructre_criteria(self, rubric_list):
        new_dict = defaultdict(list)
        for rubric_item in rubric_list:
            if 'choreographic' in rubric_item:
                new_dict['choreographic_ss'].append(rubric_item)
            elif 'step sequence' in rubric_item:
                new_dict['ss'].append(rubric_item)
            elif 'spin' in rubric_item:
                new_dict['spin'].append(rubric_item)
            else:
                new_dict['jump'].append(rubric_item)
        return new_dict

    def is_pair(self, element_a, element_b, supercategory_match=True, difficulty_match=True, finegrained_action_match=False,
                match_threshold=0.2):
        if supercategory_match:
            if element_a[3] != element_b[3]:
                # supercategory is idx 3
                return False
            if difficulty_match:
                # difficulty measured by base value
                if element_a[1] != element_b[1]:
                    return False
        elif finegrained_action_match:
            if element_a[0] != element_b[0]:
                return False

        if abs(element_a[2] - element_b[2]) <= match_threshold:
            return True
        else:
            return False

    def is_close_negative_pair(self,element_a, element_b, supercategory_match=True, difficulty_match=True,
                               finegrained_action_match=False, match_threshold=0.5):
        if supercategory_match:
            if element_a[3] != element_b[3]:
                return False
            if difficulty_match:
                # difficulty measured by base value
                if element_a[1] != element_b[1]:
                    return False

        elif finegrained_action_match:
            if element_a[0] != element_b[0]:
                return False

        if abs(element_a[2] - element_b[2]) <= match_threshold:
            return False
        else:
            return abs(element_a[2] - element_b[2])

    def get_pairs_list(self, elements_batch):
        element_batch_list = []
        for batch_idx in range(len(elements_batch[0][0])):
            for element_idx in range(len(elements_batch)):
                element = (elements_batch[element_idx][0][batch_idx], elements_batch[element_idx][1][batch_idx], elements_batch[element_idx][2][batch_idx],elements_batch[element_idx][3][batch_idx])
                element_batch_list.append(element)

        pairs_list = []
        negative_pairs_list = []
        triplets = {}
        for i, element_a in enumerate(element_batch_list):
            for j in range(i+1, len(element_batch_list)):
                if self.is_pair(element_a, element_batch_list[j]):
                    pairs_list.append((i, j))
                    if i not in triplets:
                        triplets[i]={'positive': [], 'negative': []}
                    else:
                        triplets[i]['positive'].append(j)
                elif self.is_close_negative_pair(element_a, element_batch_list[j]) > 0 :
                    negative_pairs_list.append((i, j))
                    if i not in triplets:
                        triplets[i]={'positive': [], 'negative': []}
                    else:
                        triplets[i]['negative'].append(j)
        triplet_list = []
        for key, pos_neg_dict in triplets.items():
            if len(pos_neg_dict['positive']) > 0 and len(pos_neg_dict['negative']) > 0:
                positive_item = random.sample(pos_neg_dict['positive'], k=1)[0]
                negative_item = random.sample(pos_neg_dict['negative'], k=1)[0]
                triplet_list.append((key, positive_item, negative_item))

        return element_batch_list, pairs_list, negative_pairs_list, triplet_list

    def forward(self, decoder_output_embeddings, label):
        # given a list of element indicies and positive/negative label
        if self.use_visual_text_triplet_loss:
            return self.visual_text_triplet_loss(decoder_output_embeddings, label)
        loss = torch.zeros(1).cuda()
        norm_count = 0
        losses = defaultdict(float)
        for b, element_indicies_labels in enumerate(label):
            for element_idx, action_cls, positive_label in element_indicies_labels:
                if element_idx == -1:
                    continue
                x1 = decoder_output_embeddings[b][element_idx]
                t2 = None
                if positive_label:
                    t1 = self.positive_rubric_embeddings[action_cls]
                    # broadcasting
                    # torch.ones(t1.shape[0])
                    if self.enforce_dissimilarity:
                        if self.hand_negatives:
                            # draw from hand negatives
                            t2 = self.hand_negatives_for_positive_rubric_embeddings[action_cls]
                        else:
                            t2 = self.negative_rubric_embeddings[action_cls]
                        if self.randomly_sample_dissim and random.random() > 0.5:
                            # breakpoint()
                            sampled = action_cls
                            while sampled == action_cls:
                                sampled = random.sample(self.negative_rubric_embeddings.keys(), 1)[0]
                            t3 = self.negative_rubric_embeddings[sampled]
                            t2 = torch.vstack([t2,t3])
                else:
                    t1 = self.negative_rubric_embeddings[action_cls]
                    if self.enforce_dissimilarity:
                        if self.hand_negatives:
                            t2 = self.hand_negatives_for_negative_rubric_embeddings[action_cls]
                        else:
                            t2 = self.positive_rubric_embeddings[action_cls]

                        if self.randomly_sample_dissim and random.random() > 0.5:
                            # breakpoint()
                            sampled = action_cls
                            while sampled == action_cls:
                                sampled = random.sample(self.negative_rubric_embeddings.keys(), 1)[0]
                            t3 = self.negative_rubric_embeddings[sampled]
                            t2 = torch.vstack([t2,t3])

                sim_loss = self.loss_fn(x1.unsqueeze(0), t1, torch.ones(1).cuda()).mean()
                loss += sim_loss
                losses['similarity_loss'] += sim_loss
                norm_count += 1
                if t2 is not None:
                    # breakpoint()
                    dissim_loss = self.loss_fn(x1.unsqueeze(0), t2, -1*torch.ones(1).cuda()).mean()
                    # print("Dissimilarity Loss:", dissim_loss)
                    norm_count += 1
                    loss += dissim_loss
                    losses['dissimilarity_loss'] += dissim_loss
        if 'dissimilarity_loss' in losses:
            norm_count=norm_count // 2
            losses['dissimilarity_loss']=losses['dissimilarity_loss']/(norm_count)
        else:
            losses['dissimilarity_loss']=0.0
        losses['similarity_loss'] = losses['similarity_loss'] / (norm_count)

        return  loss / norm_count, losses
    def triplet_loss(self, decoder_output_embeddings,element_batch_list, triplet_list, margin=1.0, metric='cosine'):
        loss = torch.zeros(1).cuda()
        def cosine_distance_fn(a, b):
            return 1 - F.cosine_similarity(a,b)
        def euclidean_distance_fn(a, b):
            # breakpoint()
            return torch.sqrt(((b-a)**2).sum())

        for anchor_idx, positive_idx, negative_idx in triplet_list:
            if margin < 0:
                # compute relative margin
                # if cosine, max margin can be 2, but I'll keep it as 1.5 to avoid being too harsh, for euclidean distance this will be fine
                if metric == "cosine":
                    temp_margin = min(1.5, abs(element_batch_list[negative_idx][2] - element_batch_list[anchor_idx][2]))
                else:
                    temp_margin = abs(element_batch_list[negative_idx][2] - element_batch_list[anchor_idx][2])
            else:
                temp_margin = margin
            triplet_loss = F.triplet_margin_with_distance_loss(
                decoder_output_embeddings[anchor_idx//self.num_queries][anchor_idx%self.num_queries].unsqueeze(0),
                decoder_output_embeddings[positive_idx//self.num_queries][positive_idx%self.num_queries].unsqueeze(0),
                decoder_output_embeddings[negative_idx//self.num_queries][negative_idx%self.num_queries].unsqueeze(0),
                distance_function=cosine_distance_fn if metric == 'cosine' else euclidean_distance_fn,
                margin=temp_margin
            )

            loss+=triplet_loss/len(triplet_list)
        losses={'visual_triplet_loss': loss}
        return loss, losses
    def visual_forward(self, decoder_output_embeddings, element_batch_list, pairs_list, negative_pairs_list, triplet_list, margin, use_visual_triplet_loss):
        # given a list of element indicies and positive/negative label
        loss = torch.zeros(1).cuda()
        losses = defaultdict(float)
        if use_visual_triplet_loss:
            return self.triplet_loss(decoder_output_embeddings, element_batch_list, triplet_list, margin=margin, metric=self.metric)
        for positive_pair in pairs_list:
            # element_a = element_batch_list[positive_pair[0]]
            # element_b = element_batch_list[positive_pair[1]]
            # batch size = idx // 7, element idx = idx%7
            embedding_a = decoder_output_embeddings[positive_pair[0]//self.num_queries][positive_pair[0]%self.num_queries]
            embedding_b = decoder_output_embeddings[positive_pair[1]//self.num_queries][positive_pair[1]%self.num_queries]
            sim_loss = F.cosine_embedding_loss(embedding_a.unsqueeze(0), embedding_b.unsqueeze(0), torch.ones(1).cuda())
            losses['visual_similarity_loss'] += sim_loss
        for negative_pair in negative_pairs_list:
            # element_a = element_batch_list[positive_pair[0]]
            # element_b = element_batch_list[positive_pair[1]]
            # batch size = idx // 7, element idx = idx%7
            embedding_a = decoder_output_embeddings[negative_pair[0]//self.num_queries][negative_pair[0]%self.num_queries]
            embedding_b = decoder_output_embeddings[negative_pair[1]//self.num_queries][negative_pair[1]%self.num_queries]
            # breakpoint()
            dissim_loss = F.cosine_embedding_loss(embedding_a.unsqueeze(0), embedding_b.unsqueeze(0), -1*torch.ones(1).cuda(), margin=0.5)
            losses['visual_dissimilarity_loss'] += dissim_loss
        losses['visual_similarity_loss'] = losses['visual_similarity_loss'] / len(pairs_list) if len(pairs_list) > 0 else 0
        losses['visual_dissimilarity_loss'] = losses['visual_dissimilarity_loss'] / len(negative_pairs_list) if len(negative_pairs_list) > 0 else 0
        loss += losses['visual_similarity_loss'] + losses['visual_dissimilarity_loss']
        return loss, losses
    def visual_text_triplet_loss(self, decoder_output_embeddings, label):
        loss = torch.zeros(1).cuda()

        def cosine_distance_fn(a, b):
            return 1 - F.cosine_similarity(a, b)

        def euclidean_distance_fn(a, b):
            # breakpoint()
            return torch.sqrt(((b - a) ** 2).sum())
        total_elements = 0 # decoder_output_embeddings.shape[0]*decoder_output_embeddings.shape[1]
        for b, element_indicies_labels in enumerate(label):
            for element_idx, action_cls, positive_label in element_indicies_labels:
                if element_idx == -1:
                    continue
                x1 = decoder_output_embeddings[b][element_idx]
                if positive_label:
                    t1 = self.positive_rubric_embeddings[action_cls]
                    # broadcasting
                    if self.hand_negatives:
                        # draw from hand negatives
                        t2 = self.hand_negatives_for_positive_rubric_embeddings[action_cls]
                    else:
                        t2 = self.negative_rubric_embeddings[action_cls]

                else:
                    t1 = self.negative_rubric_embeddings[action_cls]
                    if self.hand_negatives:
                        t2 = self.hand_negatives_for_negative_rubric_embeddings[action_cls]
                    else:
                        t2 = self.positive_rubric_embeddings[action_cls]
                random_idx_pos = random.randint(0, t1.shape[0]-1)
                random_idx_neg = random.randint(0, t2.shape[0]-1)
                triplet_loss = F.triplet_margin_with_distance_loss(
                    x1.unsqueeze(0),
                    t1[random_idx_pos].unsqueeze(0),
                    t2[random_idx_neg].unsqueeze(0),
                    distance_function=cosine_distance_fn if self.metric == 'cosine' else euclidean_distance_fn,
                    margin=self.margin
                )

                total_elements += 1
                loss += triplet_loss
        loss = loss / total_elements
        losses = {'visual_text_triplet_loss': loss}
        return loss, losses
