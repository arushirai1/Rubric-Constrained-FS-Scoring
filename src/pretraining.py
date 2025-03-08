"""
This module handles the pretraining phase of the model, which includes:
1. Visual-only pretraining: Learning element representations from visual features
2. Visual-text-based pretraining: Learning visually grounded rubric embeddings
3. Joint pretraining: Combining visual-only and visual-text-based representation learning
"""

import torch
import json

import numpy as np
import torch.nn.functional as F
from utils import AverageMeter
from tqdm import tqdm

def call_visual_pretraining(element_embeddings, element_info_list, loss_fn, args):
    """
    Handles the visual pretraining step by preparing the triplets and computing triplet loss.
    
    Args:
        element_embeddings: Output embeddings from the transformer decoder
        element_info_list: List containing element information like element name for each video
        loss_fn: Loss function object that computes various similarity/dissimilarity losses, including the final triplet loss
        args: Command line arguments containing training configurations
    
    Returns:
        tuple: (total_loss, dictionary of individual losses)
    """
    element_batch_list, pairs_list, negative_pairs_list, triplet_list = loss_fn.get_pairs_list(element_info_list)
    if args.vis_pos_only:
        negative_pairs_list = []
    if args.vis_neg_only:
        pairs_list = []
    if not args.visual_triplet_loss:
        triplet_list = []

    return loss_fn.visual_forward(element_embeddings, element_batch_list, pairs_list, negative_pairs_list, triplet_list,
                           args.visual_triplet_margin, args.visual_triplet_loss)

def train_epoch(epoch, model, loss_fn, regularization_fn, train_loader, optim, logger, device, args, wandb=None, visual_pretraining_only=False):
    """
    Trains the model for one epoch.
    
    Args:
        epoch: Current epoch number
        model: The transformer model being trained
        loss_fn: Loss function object
        regularization_fn: Implicit segmentation regularization loss function object
        train_loader: DataLoader containing training samples
        optim: Optimizer for updating model parameters
        logger: Logger object for tracking metrics
        device: Device (CPU/GPU) to run computations on
        args: Command line arguments
        wandb: Weights & Biases logger (optional)
        visual_pretraining_only: Whether to use only visual pretraining
    
    Returns:
        float: Average loss for the epoch
    """
    model.train()
    losses = AverageMeter('loss', logger)
    similarity_losses = AverageMeter('similarity_loss', logger)
    visual_similarity_losses = AverageMeter('visual_similarity_loss', logger)
    visual_triplet_losses = AverageMeter('visual_triplet_loss', logger)
    visual_text_triplet_losses = AverageMeter('visual_text_triplet_loss', logger)
    dissimilarity_losses = AverageMeter('dissimilarity_loss', logger)
    visual_dissimilarity_losses = AverageMeter('visual_dissimilarity_loss', logger)

    regularization_losses = regularization_fn.get_initial_avg_meter(AverageMeter, logger)
    for i, batch_values in enumerate(train_loader):
        video_feat, element_indicies, corresponding_actions, corresponding_labels, element_info_list = batch_values

        video_feat = video_feat.to(device) 
        labels=[]
        for batch_idx in range(len(element_indicies[0])):
            per_video_labels = []
            for element_idx in range(len(element_indicies)):
                per_video_labels.append((element_indicies[element_idx][batch_idx],corresponding_actions[element_idx][batch_idx], corresponding_labels[element_idx][batch_idx]))
            labels.append(per_video_labels)
        element_embeddings, attns = model.pretraining_forward(video_feat, need_weights=True)

        if visual_pretraining_only:
            visual_repr_loss, visual_losses_dict = call_visual_pretraining(element_embeddings, element_info_list, loss_fn, args)
            loss = visual_repr_loss

            if args.visual_triplet_loss:
                visual_triplet_losses.update(visual_losses_dict['visual_triplet_loss'], len(labels))
            else:
                visual_similarity_losses.update(visual_losses_dict['visual_similarity_loss'], len(labels))
                visual_dissimilarity_losses.update(visual_losses_dict['visual_dissimilarity_loss'], len(labels))
        else:
            loss, losses_dict = loss_fn(element_embeddings, labels)  # , out['embed'])
            if args.use_visual_text_triplet_loss:
                visual_text_triplet_losses.update(losses_dict['visual_text_triplet_loss'], len(labels))
            else:
                similarity_losses.update(losses_dict['similarity_loss'], len(labels))
                dissimilarity_losses.update(losses_dict['dissimilarity_loss'], len(labels))

        regularization_loss_dict = regularization_fn(attns)
        loss += args.reg_weight*regularization_loss_dict['loss']


        for key in regularization_loss_dict:
            if key == 'loss':
                # Skip the combined loss and only add individual regularization losses
                continue
            else:
                regularization_losses[key].update(regularization_loss_dict[key], len(labels))

        optim.zero_grad()
        loss.backward()
        optim.step()

        losses.update(loss, len(labels))
        print(loss)

    avg_loss = losses.done(epoch)
    avg_regularization_loss = {}
    for key in regularization_losses:
        avg_regularization_loss['avg_train_'+key] = regularization_losses[key].done(epoch)
    if wandb is not None:
        visual_losses = {}
        if visual_pretraining_only:
            if args.visual_triplet_loss:
                visual_losses = {'avg_train_visual_triplet_loss': visual_triplet_losses.done(epoch)}
            else:
                visual_losses = {'avg_train_visual_sim_loss': visual_similarity_losses.done(epoch), 'avg_train_visual_dissim_loss': visual_dissimilarity_losses.done(epoch)}
            wandb.log({'epoch': epoch,  **visual_losses})

        else:
            wandb.log({ 'epoch': epoch, 'avg_train_visual_text_triplet_loss': visual_text_triplet_losses.done(epoch), 'avg_train_sim_loss': similarity_losses.done(epoch), 'avg_train_dissim_loss': dissimilarity_losses.done(epoch)})
        wandb.log({'epoch': epoch, **avg_regularization_loss})
    return avg_loss

def test_epoch(epoch, model, loss_fn, test_loader, logger, device, args, wandb=None, visual_pretraining_only=False):
    """
    Evaluates the model for one epoch on the test set. See comment for train_epoch for more details.
    """
    model.eval()
    losses = AverageMeter('loss', logger)
    similarity_losses = AverageMeter('similarity_loss', logger)
    visual_similarity_losses = AverageMeter('visual_similarity_loss', logger)
    visual_text_triplet_losses = AverageMeter('visual_text_triplet_loss', logger)
    visual_triplet_losses = AverageMeter('visual_triplet_loss', logger)
    dissimilarity_losses = AverageMeter('dissimilarity_loss', logger)
    visual_dissimilarity_losses = AverageMeter('visual_dissimilarity_loss', logger)
    for i, batch_values in enumerate(test_loader):
        video_feat, element_indicies, corresponding_actions, corresponding_labels,element_info_list = batch_values
        video_feat = video_feat.to(device)  # (b, t, c)
        labels = []
        for batch_idx in range(len(element_indicies[0])):
            per_video_labels = []
            for element_idx in range(len(element_indicies)):
                per_video_labels.append((element_indicies[element_idx][batch_idx],
                                         corresponding_actions[element_idx][batch_idx],
                                         corresponding_labels[element_idx][batch_idx]))
            labels.append(per_video_labels)
        with torch.no_grad():
            element_embeddings, _ = model.pretraining_forward(video_feat)
            loss, losses_dict = loss_fn(element_embeddings, labels)  # , out['embed'])
            if visual_pretraining_only:
                visual_repr_loss, visual_losses_dict = call_visual_pretraining(element_embeddings, element_info_list, loss_fn,
                                                                               args)
                loss = visual_repr_loss
                if args.visual_triplet_loss:
                    visual_triplet_losses.update(visual_losses_dict['visual_triplet_loss'], len(labels))
                else:
                    visual_similarity_losses.update(visual_losses_dict['visual_similarity_loss'], len(labels))
                    visual_dissimilarity_losses.update(visual_losses_dict['visual_dissimilarity_loss'], len(labels))

            
        if args.use_visual_text_triplet_loss:
            visual_text_triplet_losses.update(losses_dict['visual_text_triplet_loss'], len(labels))
        else:
            similarity_losses.update(losses_dict['similarity_loss'], len(labels))
            dissimilarity_losses.update(losses_dict['dissimilarity_loss'], len(labels))

        losses.update(loss, len(labels))
        print("Test:", loss)


    avg_loss = losses.done(epoch)
    if wandb is not None:
        visual_losses = {}
        if visual_pretraining_only:
            if args.visual_triplet_loss:

                visual_losses = {'avg_test_visual_triplet_loss': visual_triplet_losses.done(epoch)}
            else:
                visual_losses = {'avg_test_visual_sim_loss': visual_similarity_losses.done(epoch), 'avg_test_visual_dissim_loss': visual_dissimilarity_losses.done(epoch)}
            wandb.log(
                {'epoch': epoch,  **visual_losses})

        else:
            wandb.log({'epoch': epoch, 'avg_test_sim_loss': similarity_losses.done(epoch), 'avg_test_dissim_loss': dissimilarity_losses.done(epoch), 'avg_test_visual_text_triplet_loss': visual_text_triplet_losses.done(epoch)})

    return avg_loss

def pretraining_loop(scheduler, model, loss_fn, regularization_fn, train_loader, test_loader, optim, logger, device, args, wandb, visual_train_loader, visual_test_loader):
    """
    Main pretraining loop that handles the entire pretraining process.
    
    Supports three pretraining modes:
    1. Visual-only pretraining (args.use_visual_only_pretraining_loss)
    2. Joint pretraining (args.joint_pretraining)
    3. Visual-text pretraining (default)
    
    Args:
        scheduler: Learning rate scheduler
        model: The transformer model to be trained
        loss_fn: Loss function object
        regularization_fn: Function for computing regularization losses
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
        optim: Optimizer
        logger: Logger object
        device: Device to run computations on
        args: Command line arguments
        wandb: Weights & Biases logger
        visual_train_loader: DataLoader for visual pretraining
        visual_test_loader: DataLoader for visual testing
    """
    best_test_loss=float('inf')
    if args.warmup:
        warmup = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lambda t: t / args.warmup)
    else:
        warmup = None
    pbar = tqdm(range(args.epoch))
    for epc in pbar:
        if args.warmup and epc < args.warmup:
            warmup.step()
        if args.use_visual_only_pretraining_loss:

            avg_loss = train_epoch(epc, model, loss_fn,regularization_fn, visual_train_loader, optim, logger, device, args, wandb).item()
        elif args.joint_pretraining:
            avg_loss = train_epoch(epc, model, loss_fn,regularization_fn, visual_train_loader, optim, logger, device, args, wandb, visual_pretraining_only=True).item()
            avg_loss += train_epoch(epc, model, loss_fn,regularization_fn, train_loader, optim, logger, device, args, wandb).item()

        else:
            avg_loss = train_epoch(epc, model, loss_fn,regularization_fn, train_loader, optim, logger, device, args, wandb).item()
        wandb.log({'epoch': epc, 'avg_pretraining_loss': avg_loss})

        if scheduler is not None and (args.lr_decay != 'cos' or epc >= args.warmup):
            scheduler.step()

        if args.use_visual_only_pretraining_loss:
            test_loss = test_epoch(epc, model, loss_fn, visual_test_loader, logger, device, args, wandb, visual_pretraining_only=True).item()
        elif args.joint_pretraining:
            test_loss = test_epoch(epc, model, loss_fn, visual_test_loader, logger, device, args, wandb, visual_pretraining_only=True).item()
            test_loss += test_epoch(epc, model, loss_fn, test_loader, logger, device, args, wandb).item()
            test_loss = test_loss/2 # average loss
        else:
            test_loss = test_epoch(epc, model, loss_fn, test_loader, logger, device, args, wandb).item()
        wandb.log(
            {'epoch': epc, 'avg_pretraining_loss_test': test_loss})
        if test_loss < best_test_loss:
            best_test_loss, best_epoch = test_loss, epc
            torch.save(model.state_dict(), './ckpt/' + args.model_name + '_pretraining_best.pkl')
        
        pbar.set_description('Epoch: {}\tLoss: {:.4f}\tTest Loss: {:.4f}\t'
              .format(epc, avg_loss, test_loss))

    torch.save(model.state_dict(), './ckpt/' + args.model_name + '_pretraining_final.pkl')
    print('Best Test Coef: {:.3f}\tBest Test Epoch: {}'.format(best_test_loss, best_epoch))
    if not args.test:
        wandb.finish()
