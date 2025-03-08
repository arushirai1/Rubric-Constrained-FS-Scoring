import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

def create_peak_distribution(index, length, peak_value=1, sigma=1.5):
    x = np.arange(length)
    distribution = peak_value * np.exp(-0.5 * ((x - index) / sigma)**2)
    return distribution / distribution.sum()
class SlidingPeaks:
    def __init__(self, length):
        sliding_window_of_distributions = []
        for i in range(length):
            sliding_window_of_distributions.append(torch.Tensor(create_peak_distribution(i, length)))

        self.sliding_window_of_distributions = torch.stack(sliding_window_of_distributions)
    def get_best_with_weights(self, attn_vector, weights, start_val=0, epsilon=1e-5):
        best_score = 100
        best_weighted_kl_score=100
        best_idx=-1
        for i in range(start_val, self.sliding_window_of_distributions.shape[-1]):
            kl_score=F.kl_div((attn_vector + epsilon).log(), self.sliding_window_of_distributions[i] + epsilon)
            weighted_kl_score = kl_score*(weights[i])**-1 # raise to the inverse, higher weights should lead to smaller scores, but this could lead to explosion in kl_score
            # if the loss is too high, some manual process
            if weighted_kl_score < best_weighted_kl_score:
                best_score=kl_score
                best_weighted_kl_score= weighted_kl_score
                best_idx=i
        return self.sliding_window_of_distributions[best_idx], best_idx
    def get_best(self, attn_vector, start_val=0, epsilon=1e-5):
        best_score = 100
        best_idx=-1
        for i in range(start_val, self.sliding_window_of_distributions.shape[-1]):
            kl_score=F.kl_div((attn_vector + epsilon).log(), self.sliding_window_of_distributions[i] + epsilon)
            if kl_score < best_score:
                best_score=kl_score
                best_idx=i
        return self.sliding_window_of_distributions[best_idx], best_idx
def extrema_penalty(cosine_similarities_pos, cosine_similarities_neg, temp=1, scaling_factor=1):
    extrema_penalty_pos = torch.minimum(1-cosine_similarities_pos / temp, cosine_similarities_pos / temp)
    extrema_penalty_neg = torch.minimum(1-cosine_similarities_neg / temp, cosine_similarities_neg / temp)
    total_penalty = extrema_penalty_pos.sum() + extrema_penalty_neg.sum()
    print(total_penalty)
    return total_penalty * scaling_factor

"""
class PeakOrderLoss(nn.Module):
    def __init__(self, algorithm_num):
        self.fn = self.loss_1()
    def forward(self, x):
        return self.fn(x)
    def is_violation(self, attn, i):
        for j in range(i, attn.shape[0]):
            if peak_indicies[i] > peak_indicies[j]:
                return (i,j)
        return None
    def loss_1(self, attn):
        # apply a penalty selectively if there is a violation by applying peak loss random selection
        peak_indicies = self.peaks(attn)
        penalty = 0
        for i in range(0, attn.shape[0]):
            first_violation = self.is_violation(attn, i)
            if first_violation is not None:
                # apply penalty
                penalty += 0 # TODO calcuate
            # randomly select peak between the next non-violating peak; randomly select all peaks in betweek
        return penalty
    def constraint_satisfier_alg_best(self, peaks):
        # use dynamic programming to balance ideal peaks while satisfying order
        return 0
    def constraint_satisfier_alg_greedy(self, peaks):
        # peaks is a 2D list (queries, peaks)
        # greedy algorithm
        solution = [-1]
        for query_i, peak_indicies in enumerate(peaks):
            selected_idx = -1
            for peak_idx in peak_indicies:
                if peak_idx > solution[-1]:
                    selected_idx = peak_idx
                    break
            if selected_idx == -1:
                # can't solve
                return None
            else:
                solution.append(selected_idx)
        return solution[1:]
    def loss_2(self, attn):
        # select peak for peak loss such that it satisfies constraints
        # find all possible peaks using mean shift clustering, I think I have this code somewhere in some notebook
        peaks = self.get_all_peaks(attn)
        # find a solution through the peaks that satisfies constraints, if it's impossible then call loss_1 as fall back
        peaks_reduced = self.constraint_satisfier_alg_greedy(peaks)
        target_gaussians=[]
        for attn_idx, peak_idx in enumerate(peaks_reduced):
            target_gaussians.append(torch.Tensor(create_peak_distribution(peak_idx, attn.shape[-1])))

        target_gaussian = torch.stack(target_gaussians).view(-1, attn.shape[1], attn.shape[2]).cuda()
        attn += epsilon
        target_gaussian += epsilon
        return F.kl_div(attn.log(), target_gaussian, reduction='sum')

    def loss_3(self, attn):
        # apply sliding window loss and limit to those that don't violate order
        peak_loss = 0
        start_idx = 0
        for query_idx in range(attn.shape[0]):
            if start_idx >= attn.shape[0]:
                raise Exception("Previously selected value caused this to be unsolvable")
                # apply loss 1 as fall back
            sliding_peak_loss, start_idx = self.peak_obj.get_best(attn_vector, start_val=start_idx+1, epsilon=1e-5)
            peak_loss += sliding_peak_loss
        return peak_loss

    def loss_4(self, attn):
        # this loss will apply order prior in addition to the sliding peaks window, this prior probability will influence the selection of the peak used for the slidin peak lsos
        # how to represent the prior probabilities
        # how should weights be determined?
        return 0
"""
class RegularizationLossFn(nn.Module):
    def __init__(self, ortho_loss, peak_loss, sliding_peak_loss, clip_l1_norm, args):
        super(RegularizationLossFn, self).__init__()
        self.ortho_loss = ortho_loss
        self.peak_loss = peak_loss
        self.sliding_peak_loss = sliding_peak_loss
        self.clip_l1_norm = clip_l1_norm
        if sliding_peak_loss:
            self.peak_obj = SlidingPeaks(args.clip_num)
    def get_initial_avg_meter(self, AverageMeter, logger):
        loss_meter_dicts = {}
        if self.ortho_loss:
            loss_meter_dicts['ortho_loss']=AverageMeter('ortho_loss', logger)
        if self.peak_loss:
            loss_meter_dicts['peak_loss'] = AverageMeter('peak_loss', logger)
        return loss_meter_dicts
    def forward(self, attns):
        # given a list of element indicies and positive/negative label
        loss_dict={}
        loss_dict['loss']=torch.zeros(1).cuda()
        attns = attns[-1]

        if self.peak_loss:
            epsilon = 1e-5
            peak_lambda = 0.005
            highest_attention_indices = torch.argmax(attns, axis=2)
            target_gaussians = []
            for batch_num, peak_indicies in enumerate(highest_attention_indices):
                for attn_idx, peak_idx in enumerate(peak_indicies):
                    if self.sliding_peak_loss:
                        target_gaussians.append(self.peak_obj.get_best(attns[batch_num][attn_idx].cpu()))
                    else:
                        target_gaussians.append(
                            torch.Tensor(create_peak_distribution(peak_idx.item(), attns.shape[-1])))
            target_gaussian = torch.stack(target_gaussians).view(-1, attns.shape[1], attns.shape[2]).cuda()
            attns += epsilon
            target_gaussian += epsilon
            peak_loss = peak_lambda * F.kl_div(attns.log(), target_gaussian, reduction='sum')
            loss_dict['peak_loss']=peak_loss
            loss_dict['loss'] += peak_loss

        if self.ortho_loss:
            ortho_lambda = 20000.0 # set for pretraining
            dot_prod_matrix = torch.matmul(attns, attns.transpose(2, 1))

            othogonality_matrix = torch.eye(attns.shape[1]).repeat(attns.shape[0], 1).view(attns.shape[0], attns.shape[1], attns.shape[1])\
                .cuda()  # zero with other vectors
            mask = torch.ones(othogonality_matrix.shape)
            mask = (mask - torch.eye(mask.shape[-1])).cuda()
            ortho_loss = ortho_lambda * F.mse_loss(dot_prod_matrix * mask, othogonality_matrix * mask)
            loss_dict['ortho_loss']=ortho_loss

            loss_dict['loss'] += ortho_loss
        return loss_dict
