import json

import numpy as np
from scipy.stats import spearmanr
import pdb
import torch
import torch.nn.functional as F
from utils import AverageMeter

def train_epoch(epoch, model, loss_fn, regularization_fn, train_loader, optim, logger, device, args, wandb=None):
    """
    Trains the model for one epoch.
    
    Args:
        epoch: Current epoch number
        model: The transformer model being trained
        loss_fn: Loss function object for computing main losses
        regularization_fn: Function for computing regularization losses (L2, orthogonality, peak, etc.)
        train_loader: DataLoader containing training samples
        optim: Optimizer for updating model parameters
        logger: Logger object for tracking metrics
        device: Device (CPU/GPU) to run computations on
        args: Command line arguments
        wandb: Weights & Biases logger (optional)
    
    Returns:
        tuple: (average_loss, correlation_coefficient)
    """
    model.train()
    preds = np.array([])
    labels = np.array([])

    # Initialize loss trackers
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

    # Initialize regularization loss trackers if regularization_fn is provided
    regularization_losses = {}
    if regularization_fn is not None:
        regularization_losses = regularization_fn.get_initial_avg_meter(AverageMeter, logger)

    for i, batch_values in enumerate(train_loader):
        # Handle different input formats based on configuration
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

        # Move data to device
        if not args.action_rubric_mask:
            pos_action_rubric_mask = None
            neg_action_rubric_mask = None
        else:
            pos_action_rubric_mask = pos_action_rubric_mask.to(device)
            neg_action_rubric_mask = neg_action_rubric_mask.to(device)
        video_feat = video_feat.to(device)
        base_values = base_values.to(device)
        if args.dataset_name != 'rg':
            gt_goes = gt_goes.to(device)
        label = label.float().to(device)

        # Forward pass
        need_weights = regularization_fn is not None
        if args.clip_embedding_path is not None:
            out = model(video_feat, base_values, pos_action_rubric_mask, neg_action_rubric_mask, 
                       args.rescaling, need_weights, clip_vision_inject=clip_vision_inject)
        else:
            out = model(video_feat, base_values, rescaling=args.rescaling, 
                       need_weights=need_weights, gdlt=args.gdlt, 
                       clip_vision_inject=clip_vision_inject)
        
        pred = out['output']

        # Compute main loss
        if args.gdlt:
            loss, mse, tri = loss_fn(pred, label, out['embed'])
        else:
            loss, mse, tri = loss_fn(pred, label)

        # Add regularization losses if regularization_fn is provided
        if regularization_fn is not None:
            regularization_loss_dict = regularization_fn(out)
            loss += args.reg_weight * regularization_loss_dict['loss']

            # Update regularization loss trackers
            for key in regularization_loss_dict:
                if key == 'loss':
                    continue
                regularization_losses[key].update(regularization_loss_dict[key], label.shape[0])

        if 'losses' in out:
            for loss_name in out['losses']:
                loss += out['losses'][loss_name]
                if 'l1' in loss_name:
                    clip_l1_losses.update(out['losses'][loss_name], label.shape[0])

        # Optimization step
        optim.zero_grad()
        loss.backward()
        optim.step()
        # save gradients
        if args.grad_stats:
            gradient_stats_by_epoch.append(get_grad_statistics(model))
            # with open()
        losses.update(loss, label.shape[0])
        mse_losses.update(mse, label.shape[0])
        tri_losses.update(tri, label.shape[0])

        # Store predictions for correlation computation
        if len(preds) == 0:
            preds = pred.cpu().detach().numpy()
            labels = label.cpu().detach().numpy()
        else:
            preds = np.concatenate((preds, pred.cpu().detach().numpy()), axis=0)
            labels = np.concatenate((labels, label.cpu().detach().numpy()), axis=0)

    # Compute correlation coefficient
    coef, _ = spearmanr(preds, labels)
    if logger is not None:
        logger.add_scalar('train coef', coef, epoch)

    # Log metrics
    avg_loss = losses.done(epoch)
    other_losses_dict['train_ortho_loss']=ortho_losses.done(epoch)
    other_losses_dict['train_peak_loss']=peak_losses.done(epoch)
    other_losses_dict['train_bv_loss']=bv_losses.done(epoch)
    other_losses_dict['train_extrema_penalty_loss']=extrema_penalty_losses.done(epoch)
    if args.clip_l1_norm:
        other_losses_dict['train_clip_l1_loss']=clip_l1_losses.done(epoch)

    # Finish adding regularization losses to logging dict if available
    if regularization_fn is not None:
        for key in regularization_losses:
            other_losses_dict['train_' + key] = regularization_losses[key].done(epoch)

    if wandb is not None:
        wandb.log({
            'train_coef': coef, 
            'epoch': epoch, 
            'avg_loss': avg_loss, 
            **other_losses_dict
        })
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
