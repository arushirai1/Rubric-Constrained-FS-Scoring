import numpy as np
import torch
from scipy.stats import spearmanr
from torch import nn

def test_epoch(epoch, model, test_loader, logger, device, args, wandb=None):
    mse_loss = nn.MSELoss().to(device)
    model.eval()

    preds = np.array([])
    labels = np.array([])
    tol_loss, tol_sample = 0, 0

    feats = []

    with torch.no_grad():
        for i, batch_values in enumerate(test_loader):
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
                clip_vision_inject=None

            if not args.action_rubric_mask:
                pos_action_rubric_mask = None
                neg_action_rubric_mask = None
            else:
                pos_action_rubric_mask = pos_action_rubric_mask.to(device)
                neg_action_rubric_mask = neg_action_rubric_mask.to(device)
            video_feat = video_feat.to(device)
            base_values = base_values.to(device)
            label = label.float().to(device)
            if args.clip_embedding_path is not None:
                out = model(video_feat, base_values, pos_action_rubric_mask, neg_action_rubric_mask, args.rescaling, clip_vision_inject=clip_vision_inject)
            else:
                out = model(video_feat, base_values, rescaling=args.rescaling, gdlt=args.gdlt,clip_vision_inject=clip_vision_inject)
            pred = out['output']

            if 'encode' in out.keys() and out['encode'] is not None:
                feats.append(out['encode'].mean(dim=1).cpu().detach().numpy())
                # feats.append(out['embed'].cpu().detach().numpy())
            if args.gdlt:
                loss = mse_loss(pred*42.81, label*42.81) # this is to remove the normalization factor of 42.81 (see dataset.py, you must have normalization on to reproduct gdlt results)
            else:
                loss = mse_loss(pred, label)
            tol_loss += (loss.item() * label.shape[0])
            tol_sample += label.shape[0]

            if len(preds) == 0:
                preds = pred.cpu().detach().numpy()
                labels = label.cpu().detach().numpy()
            else:
                preds = np.concatenate((preds, pred.cpu().detach().numpy()), axis=0)
                labels = np.concatenate((labels, label.cpu().detach().numpy()), axis=0)
    # print(preds)
    avg_coef, _ = spearmanr(preds, labels)
    avg_loss = float(tol_loss) / float(tol_sample)
    if logger is not None:
        logger.add_scalar('Test coef', avg_coef, epoch)
        logger.add_scalar('Test loss', avg_loss, epoch)
    if wandb is not None:
        wandb.log({'test_coef' : avg_coef, 'epoch': epoch, 'test_loss': avg_loss})
    # print(preds.tolist())
    # print(labels.tolist())
    return avg_loss, avg_coef
