import json

import numpy as np
from scipy.stats import spearmanr
import pdb
import torch
import torch.nn.functional as F
from utils import AverageMeter

def create_peak_distribution(index, length, peak_value=1, sigma=1.5):
    x = np.arange(length)
    distribution = peak_value * np.exp(-0.5 * ((x - index) / sigma)**2)
    return distribution / distribution.sum()

def get_grad_statistics(model):
    grad_stats = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            grad_stats[name] = {
                'mean': parameter.grad.mean().item(),
                'std': parameter.grad.std().item()
            }
    return grad_stats

class SlidingPeaks:
    def __init__(self, length):
        sliding_window_of_distributions = []
        for i in range(length):
            sliding_window_of_distributions.append(torch.Tensor(create_peak_distribution(i, length)))

        self.sliding_window_of_distributions = torch.stack(sliding_window_of_distributions)
    def get_best(self, attn_vector, epsilon=1e-5):
        best_score = 100
        best_idx=-1
        for i in range(self.sliding_window_of_distributions.shape[-1]):
            kl_score=F.kl_div((attn_vector + epsilon).log(), self.sliding_window_of_distributions[i] + epsilon)
            if kl_score < best_score:
                best_score=kl_score
                best_idx=i
        return self.sliding_window_of_distributions[best_idx]
def extrema_penalty(cosine_similarities_pos, cosine_similarities_neg, temp=1, scaling_factor=1):
    extrema_penalty_pos = torch.minimum(1-cosine_similarities_pos / temp, cosine_similarities_pos / temp)
    extrema_penalty_neg = torch.minimum(1-cosine_similarities_neg / temp, cosine_similarities_neg / temp)
    total_penalty = extrema_penalty_pos.sum() + extrema_penalty_neg.sum()
    print(total_penalty)
    return total_penalty * scaling_factor
def train_epoch(epoch, model, loss_fn, train_loader, optim, logger, device, args, wandb=None):
    model.train()
    preds = np.array([])
    labels = np.array([])

    losses = AverageMeter('loss', logger)
    peak_losses = AverageMeter('peak_loss', logger)
    bv_losses = AverageMeter('bv_loss', logger)
    ortho_losses = AverageMeter('ortho_loss', logger)
    mse_losses = AverageMeter('mse', logger)
    tri_losses = AverageMeter('tri', logger)
    extrema_penalty_losses = AverageMeter('extrema_penalty', logger)
    clip_l1_losses = AverageMeter('train_clip_l1_losses', logger)
    gradient_stats_by_epoch=[]
    other_losses_dict = {}
    if args.peak_loss:
        peak_obj = SlidingPeaks(args.clip_num)
    for i, batch_values in enumerate(train_loader):
        
        if args.vision_vlm_inject:
            if args.action_rubric_mask:
                video_feat, clip_vision_inject, gt_goes, base_values, label, pos_action_rubric_mask, neg_action_rubric_mask = batch_values
            else:
                video_feat, clip_vision_inject, gt_goes, base_values, label = batch_values
            clip_vision_inject = clip_vision_inject.to(device)
        else:
            if args.action_rubric_mask:
                video_feat, gt_goes, base_values, label, pos_action_rubric_mask, neg_action_rubric_mask = batch_values
            else:
                if args.dataset_name == 'rg':
                    video_feat, label = batch_values
                    base_values = torch.zeros(1)
                else:
                    video_feat, gt_goes, base_values, label = batch_values
            clip_vision_inject = None

        if not args.action_rubric_mask:
            pos_action_rubric_mask = None
            neg_action_rubric_mask = None
        else:
            pos_action_rubric_mask=pos_action_rubric_mask.to(device)
            neg_action_rubric_mask=neg_action_rubric_mask.to(device)
        video_feat = video_feat.to(device)      # (b, t, c)
        base_values = base_values.to(device)
        if args.dataset_name != 'rg':
            gt_goes = gt_goes.to(device)
        label = label.float().to(device)
        if args.clip_embedding_path is not None:
            out = model(video_feat, base_values, pos_action_rubric_mask, neg_action_rubric_mask, args.rescaling, args.l2_regularization or args.peak_loss or args.ortho_loss, clip_vision_inject=clip_vision_inject)
        else:
            out = model(video_feat, base_values,  rescaling=args.rescaling, need_weights= args.l2_regularization or args.peak_loss or args.ortho_loss, gdlt=args.gdlt, clip_vision_inject=clip_vision_inject)
        pred = out['output']
        # print(pred)
        if args.gdlt:
            loss, mse, tri = loss_fn(pred, label, out['embed'])
        else:
            loss, mse, tri = loss_fn(pred, label)#, out['embed'])
        if args.extrema_penalty:

            total_extrema_loss=extrema_penalty(out['pos_rubric_cos_sims'], out['neg_rubric_cos_sims'])
            loss+=total_extrema_loss
            extrema_penalty_losses.update(total_extrema_loss, label.shape[0])

        if 'losses' in out:
            for loss_name in out['losses']:
                loss += out['losses'][loss_name]
                if 'l1' in loss_name:
                    clip_l1_losses.update(out['losses'][loss_name], label.shape[0])
        if args.predict_goe:
            loss += F.mse_loss(out["goe_predictions"], gt_goes)
        if args.l2_regularization:
            attns=out['attns']
            l2_lambda=1.0
            l2_regularization = torch.sum(torch.diag(torch.matmul(attns[-1].reshape(-1,128), attns[-1].reshape(-1,128).t())))
            loss += l2_lambda*l2_regularization
        if args.predict_bv:
            bv_lambda = 1# 0.5
            bv_loss = bv_lambda * F.mse_loss(out["base_value_predictions"], base_values)
            if epoch < 100:
                loss = bv_loss
            else:
                loss += bv_lambda * bv_loss

            bv_losses.update(bv_loss, label.shape[0])
        if args.peak_loss:
            epsilon=1e-5
            peak_lambda=0.005
            attns = out['attns'][-1]
            highest_attention_indices = torch.argmax(attns, axis=2)
            target_gaussians=[]
            for batch_num, peak_indicies in enumerate(highest_attention_indices):
                for attn_idx, peak_idx in enumerate(peak_indicies):
                    if args.sliding_peak_loss:
                        target_gaussians.append(peak_obj.get_best(attns[batch_num][attn_idx].cpu()))
                    else:
                        target_gaussians.append(torch.Tensor(create_peak_distribution(peak_idx.item(), attns.shape[-1])))
            target_gaussian = torch.stack(target_gaussians).view(-1, attns.shape[1],attns.shape[2]).to(device)
            attns+=epsilon
            target_gaussian +=epsilon
            peak_loss = peak_lambda*F.kl_div(attns.log(), target_gaussian, reduction='sum')
            loss += peak_loss
            peak_losses.update(peak_loss, label.shape[0])

        if args.ortho_loss:
            ortho_lambda=100000.0
            attn_shape=out['attns'][-1].shape
            attns = out['attns'][-1]
            dot_prod_matrix = torch.matmul(attns, attns.transpose(2,1))

            othogonality_matrix = torch.eye(attns.shape[1]).repeat(attns.shape[0],1).view(attns.shape[0],attns.shape[1], attns.shape[1]).to(device) # zero with other vectors
            # attns = out['attns'][-1].view(-1, attn_shape[-1])
            # dot_prod_matrix = torch.matmul(attns, attns.t())
            # dot_prod_matrix = dot_prod_matrix.view(attn_shape[0], attn_shape[1], -1)
            # othogonality_matrix = torch.eye(attns.shape[1]).repeat(attns.shape[0], 1).view(attns.shape[0], 7,
            #                                                                                -1)  # zero with other vectors
            # mask = torch.ones(othogonality_matrix.shape)
            # mask = mask - torch.eye(mask.shape[-1])
            # mask = F.pad(mask, (0, dot_prod_matrix.shape[-1] - attn_shape[1]))
            #
            # loss += ortho_lambda * F.mse_loss(dot_prod_matrix * mask, othogonality_matrix.to(device))
            mask = torch.ones(othogonality_matrix.shape)
            mask = (mask - torch.eye(mask.shape[-1])).to(device)
            # breakpoint()
            ortho_loss = ortho_lambda * F.mse_loss(dot_prod_matrix* mask, othogonality_matrix* mask)
            loss+= ortho_loss
            ortho_losses.update(ortho_loss, label.shape[0])

        # breakpoint()
        optim.zero_grad()
        loss.backward()
        # pdb.set_trace()
        optim.step()
        # save gradients
        if args.grad_stats:
            gradient_stats_by_epoch.append(get_grad_statistics(model))
            # with open()
        losses.update(loss, label.shape[0])
        mse_losses.update(mse, label.shape[0])
        tri_losses.update(tri, label.shape[0])

        if len(preds) == 0:
            preds = pred.cpu().detach().numpy()
            labels = label.cpu().detach().numpy()
        else:
            preds = np.concatenate((preds, pred.cpu().detach().numpy()), axis=0)
            labels = np.concatenate((labels, label.cpu().detach().numpy()), axis=0)

    coef, _ = spearmanr(preds, labels)
    if logger is not None:
        logger.add_scalar('train coef', coef, epoch)

    avg_loss = losses.done(epoch)
    other_losses_dict['train_ortho_loss']=ortho_losses.done(epoch)
    other_losses_dict['train_peak_loss']=peak_losses.done(epoch)
    other_losses_dict['train_bv_loss']=bv_losses.done(epoch)
    other_losses_dict['train_extrema_penalty_loss']=extrema_penalty_losses.done(epoch)
    if args.clip_l1_norm:
        other_losses_dict['train_clip_l1_loss']=clip_l1_losses.done(epoch)
    if wandb is not None:
        wandb.log({'train_coef' : coef, 'epoch': epoch, 'avg_loss': avg_loss, **other_losses_dict})
    if args.grad_stats:
        if epoch == 0:
            permission_type='w'
        else:
            permission_type='a'

        with open(f'./grad_stats/{args.model_name}.txt', permission_type) as f:
            save_str = json.dumps(gradient_stats_by_epoch)
            f.write(save_str+'\n')

    mse_losses.done(epoch)
    tri_losses.done(epoch)
    return avg_loss, coef
