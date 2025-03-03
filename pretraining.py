import torch
import json

import numpy as np
import torch.nn.functional as F
from utils import AverageMeter
from tqdm import tqdm
def call_visual_pretraining(decoder_embeddings, element_info_list, loss_fn, args):
    element_batch_list, pairs_list, negative_pairs_list, triplet_list = loss_fn.get_pairs_list(element_info_list)
    if args.vis_pos_only:
        negative_pairs_list = []
    if args.vis_neg_only:
        pairs_list = []
    if not args.visual_triplet_loss:
        triplet_list = []
    # breakpoint()

    return loss_fn.visual_forward(decoder_embeddings, element_batch_list, pairs_list, negative_pairs_list, triplet_list,
                           args.visual_triplet_margin, args.visual_triplet_loss)
def train_epoch(epoch, model, loss_fn, regularization_fn, train_loader, optim, logger, device, args, wandb=None, visual_pretraining_only=False):
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
        # breakpoint()

        # print(f"Total Length of Positive Pairs: {len(pairs_list)}; Total Length of Negative Pairs: {len(negative_pairs_list)}")
        video_feat = video_feat.to(device)      # (b, t, c)
        labels=[]
        for batch_idx in range(len(element_indicies[0])):
            per_video_labels = []
            for element_idx in range(len(element_indicies)):
                per_video_labels.append((element_indicies[element_idx][batch_idx],corresponding_actions[element_idx][batch_idx], corresponding_labels[element_idx][batch_idx]))
            labels.append(per_video_labels)
        decoder_embeddings, attns = model.pretraining_forward(video_feat, need_weights=True)

        if visual_pretraining_only:
            visual_repr_loss, visual_losses_dict = call_visual_pretraining(decoder_embeddings, element_info_list, loss_fn, args)
            # breakpoint()
            loss = visual_repr_loss

            if args.visual_triplet_loss:
                visual_triplet_losses.update(visual_losses_dict['visual_triplet_loss'], len(labels))
            else:
                visual_similarity_losses.update(visual_losses_dict['visual_similarity_loss'], len(labels))
                visual_dissimilarity_losses.update(visual_losses_dict['visual_dissimilarity_loss'], len(labels))
        else:
            loss, losses_dict = loss_fn(decoder_embeddings, labels)  # , out['embed'])
            if args.use_visual_text_triplet_loss:
                visual_text_triplet_losses.update(losses_dict['visual_text_triplet_loss'], len(labels))
            else:
                similarity_losses.update(losses_dict['similarity_loss'], len(labels))
                dissimilarity_losses.update(losses_dict['dissimilarity_loss'], len(labels))

        regularization_loss_dict = regularization_fn(attns)
        # loss =visual_repr_loss # temp
        loss += args.reg_weight*regularization_loss_dict['loss']


        for key in regularization_loss_dict:
            if key == 'loss':
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
    # breakpoint()
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
            decoder_embeddings, _ = model.pretraining_forward(video_feat)
            loss, losses_dict = loss_fn(decoder_embeddings, labels)  # , out['embed'])
            if visual_pretraining_only:
                # breakpoint()
                visual_repr_loss, visual_losses_dict = call_visual_pretraining(decoder_embeddings, element_info_list, loss_fn,
                                                                               args)
                loss = visual_repr_loss
                if args.visual_triplet_loss:
                    visual_triplet_losses.update(visual_losses_dict['visual_triplet_loss'], len(labels))
                else:
                    visual_similarity_losses.update(visual_losses_dict['visual_similarity_loss'], len(labels))
                    visual_dissimilarity_losses.update(visual_losses_dict['visual_dissimilarity_loss'], len(labels))

            # visual_repr_loss, visual_losses_dict = loss_fn.visual_forward(decoder_embeddings, element_batch_list,
            #                                                               pairs_list, negative_pairs_list, triplet_list, args.visual_triplet_margin)
        if args.use_visual_text_triplet_loss:
            visual_text_triplet_losses.update(losses_dict['visual_text_triplet_loss'], len(labels))
        else:
            similarity_losses.update(losses_dict['similarity_loss'], len(labels))
            dissimilarity_losses.update(losses_dict['dissimilarity_loss'], len(labels))

        # loss =visual_repr_loss # TEMP


        losses.update(loss, len(labels))
        print("Test:", loss)


    avg_loss = losses.done(epoch)
    # breakpoint()
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
    best_test_loss=float('inf')
    if args.warmup:
        warmup = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lambda t: t / args.warmup)
    else:
        warmup = None
    pbar = tqdm(range(args.epoch))
    # test_loss=200 # REMOVE AFTER LEARNING RATE HAS BEEN PICKED
    for epc in pbar:
        if args.warmup and epc < args.warmup:
            warmup.step()
        if args.use_visual_only_pretraining_loss:

            avg_loss = train_epoch(epc, model, loss_fn,regularization_fn, visual_train_loader, optim, logger, device, args, wandb).item()
        elif args.joint_pretraining:
            # breakpoint()
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
        # test_loss= 0 # delete later
        # if epc % 10 == 0:
        #     torch.save(model.state_dict(), './ckpt/' + args.model_name + '_pretraining_best.pkl') # DELETE later
        pbar.set_description('Epoch: {}\tLoss: {:.4f}\tTest Loss: {:.4f}\t'
              .format(epc, avg_loss, test_loss))
        if epc == args.epoch - 1:
            final_train_loss, final_test_loss = \
                avg_loss, test_loss
    torch.save(model.state_dict(), './ckpt/' + args.model_name + '_pretraining_final.pkl')
    print('Best Test Coef: {:.3f}\tBest Test Epoch: {}'.format(best_test_loss, best_epoch))
    if not args.test:
        wandb.finish()
