import torch
import numpy as np
import options
from datasets import RGDataset, FisVDataset, FisVPretrainingDataset, FS800Dataset, FS800PretrainingDataset
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from models import model, loss, pretraining_loss
import os
from tqdm import tqdm
from torch import nn

import train
import pretraining
from test import test_epoch
import wandb

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def setup_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_optim(model, args):
    if args.optim == 'sgd':
        optim = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optim == 'adam':
        optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim == 'adamw':
        optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim == 'rmsprop':
        optim = torch.optim.RMSprop(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        raise Exception("Unknown optimizer")
    return optim


def get_scheduler(optim, args):
    if args.lr_decay is not None:
        if args.lr_decay == 'cos':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optim, T_max=args.epoch - args.warmup, eta_min=args.lr * args.decay_rate)
        elif args.lr_decay == 'multistep':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[args.epoch - 30], gamma=args.decay_rate)
        else:
            raise Exception("Unknown Scheduler")
    else:
        scheduler = None
    return scheduler


if __name__ == '__main__':
    args = options.parser.parse_args()
    setup_seed(0)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # set dataset class and pretraining dataset class
    dataset_class = None
    if args.dataset_name == 'fs800':
        dataset_class = FS800Dataset
        pretraining_dataset_class = FS800PretrainingDataset 
    elif args.dataset_name == 'rg':
        dataset_class = RGDataset
    else:
        dataset_class =  FisVDataset
        pretraining_dataset_class = FisVPretrainingDataset 

    # load pretraining data
    if args.pretraining:
        pretraining_data = pretraining_dataset_class(args.video_path, args.train_label_path, vid_id_to_element_list_path=args.vid_id_to_element_list_path, clip_num=args.clip_num,
                    score_type=args.score_type, action_masks=args.action_rubric_mask, gdlt=args.gdlt,
                    vision_vlm_inject_path=args.vision_vlm_inject_path, pretraining_threshold_pos=args.pretraining_threshold_pos, pretraining_threshold_neg=args.pretraining_threshold_neg)
        train_data = pretraining_data
        visual_train_loader=None
        if args.joint_pretraining or args.use_visual_only_pretraining_loss:
            visual_pretraining_data = pretraining_dataset_class(args.video_path, args.train_label_path, vid_id_to_element_list_path=args.vid_id_to_element_list_path, clip_num=args.clip_num,
                                                      score_type=args.score_type, action_masks=args.action_rubric_mask,
                                                      gdlt=args.gdlt,
                                                      vision_vlm_inject_path=args.vision_vlm_inject_path,
                                                      pretraining_threshold_pos=0.0,#0.01,
                                                      pretraining_threshold_neg=0.0)#-0.01)
            visual_train_loader = DataLoader(train_data, batch_size=args.batch, drop_last=args.drop_last, shuffle=True,
                                      num_workers=args.n_workers)

    # load train data
    elif args.dataset_name != 'rg':
        train_data = dataset_class(args.video_path, args.train_label_path, vid_id_to_element_list_path=args.vid_id_to_element_list_path, clip_num=args.clip_num,
                               score_type=args.score_type, action_masks=args.action_rubric_mask, gdlt=args.gdlt, vision_vlm_inject_path=args.vision_vlm_inject_path)
    else:
        train_data = dataset_class(args.video_path, args.train_label_path, clip_num=args.clip_num, action_type=args.action_type, score_type=args.score_type, train=True)
    train_loader = DataLoader(train_data, batch_size=args.batch, drop_last=args.drop_last, shuffle=True, num_workers=args.n_workers)
    print(len(train_data))

    # load test data
    if args.pretraining:
        test_data = pretraining_dataset_class(args.video_path, args.test_label_path, clip_num=args.clip_num,
                              score_type=args.score_type, train=False, vid_id_to_element_list_path=args.vid_id_to_element_list_path, action_masks=args.action_rubric_mask, gdlt=args.gdlt, vision_vlm_inject_path=args.vision_vlm_inject_path, pretraining_threshold_pos=args.pretraining_threshold_pos, pretraining_threshold_neg=args.pretraining_threshold_neg)
        batch_size = args.batch
        visual_test_loader = None
        if args.joint_pretraining or args.use_visual_only_pretraining_loss:
            visual_pretraining_test_data = pretraining_dataset_class(args.video_path, args.test_label_path, clip_num=args.clip_num,
                              score_type=args.score_type, train=False, vid_id_to_element_list_path=args.vid_id_to_element_list_path, action_masks=args.action_rubric_mask, gdlt=args.gdlt, vision_vlm_inject_path=args.vision_vlm_inject_path, 
                              pretraining_threshold_pos=0.0,#0.01
                              pretraining_threshold_neg=0.0)#-0.01)

            visual_test_loader =  DataLoader(visual_pretraining_test_data, batch_size=batch_size, shuffle=False, num_workers=args.n_workers)

    elif args.dataset_name != 'rg':
        test_data = dataset_class(args.video_path, args.test_label_path, clip_num=args.clip_num,
                            score_type=args.score_type, train=False, vid_id_to_element_list_path=args.vid_id_to_element_list_path, action_masks=args.action_rubric_mask, gdlt=args.gdlt, vision_vlm_inject_path=args.vision_vlm_inject_path)
        batch_size = 1
    else:
        test_data = dataset_class(args.video_path, args.test_label_path, clip_num=args.clip_num, action_type=args.action_type, score_type=args.score_type, train=False)
        batch_size = 1
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=args.n_workers)
    print("Test Data Len:", len(test_data))
    print('=============Load dataset successfully=============')

    # load model
    if args.clip_embedding_path is not None:
        if args.n_decoder > 0:
            model = model.CLIP_GDLT_W_Decoder(args.in_dim, args.n_head, args.n_encoder,
                               args.n_decoder, args.n_query, args.dropout, args.clip_embedding_path, args.clip_l1_norm, args.clip_sigmoid, args.clip_relu, args.vision_vlm_inject, args.restrict_first_quad, args.rubric_simplified, is_rg = args.dataset_name == 'rg').to(device)
        else:
            model = model.CLIP_GDLT(args.in_dim, args.n_head, args.n_encoder, args.dropout, args.clip_embedding_path).to(device)
    else:
        model = model.GDLT(args.in_dim, args.hidden_dim, args.n_head, args.n_encoder,
                           args.n_decoder, args.n_query, args.dropout, args.predict_bv, args.predict_deductions, gdlt=args.gdlt).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters in {args.model_name} model: {num_params}")

    # load regularization loss function
    from models.regularization_loss import RegularizationLossFn
    if args.ortho_loss or args.peak_loss or args.sliding_peak_loss:
        regularization_fn = RegularizationLossFn(args.ortho_loss, args.peak_loss, args.sliding_peak_loss, args.clip_l1_norm, args)
    else:
        regularization_fn = None

    # load pretraining loss function
    if args.pretraining:
        loss_fn = pretraining_loss.PretrainingLossFn('clip', aggregation_method = args.pretraining_agg, margin = args.margin, enforce_dissimilarity=args.enforce_dissimilarity, randomly_sample_dissim= args.randomly_sample_dissim, metric = args.dist_metric, simplified=args.rubric_simplified, use_text_prompt=args.rubric_use_text_prompt, hand_negatives=args.rubric_hand_negatives, visual_text_triplet_loss=args.use_visual_text_triplet_loss)
    else:
        loss_fn = loss.LossFun(args.alpha, args.margin)

    # load train function
    train_fn = train.train_epoch

    # load checkpoint
    if args.ckpt is not None:
        checkpoint = torch.load('./ckpt/' + args.ckpt + '.pkl')
        if args.fisvtofs800_pretrain:
            # Load with strict=False to allow partial loading
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
            
            # Print the missing and unexpected keys
            if missing_keys:
                print(f"Missing keys: {missing_keys}")
            if unexpected_keys:
                print(f"Unexpected keys: {unexpected_keys}")
        else:
            model.load_state_dict(checkpoint)
    print('=============Load model successfully=============')

    print(args)

    # test mode
    if args.test:
        test_loss, coef = test_epoch(0, model, test_loader, None, device, args)
        print('Test Loss: {:.4f}\tTest Coef: {:.3f}'.format(test_loss, coef))
        raise SystemExit

    # record
    if not os.path.exists("./ckpt/"):
        os.makedirs("./ckpt/")
    if not os.path.exists("./logs/" + args.model_name):
        os.makedirs("./logs/" + args.model_name)
    logger = SummaryWriter(os.path.join('./logs/', args.model_name))
    best_coef, best_epoch = -1, -1
    final_train_loss, final_train_coef, final_test_loss, final_test_coef = 0, 0, 0, 0

    # train
    optim = get_optim(model, args)
    scheduler = get_scheduler(optim, args)
    wandb.init(project="figure_skating_scoring", entity="arai4")
    wandb.config.update(args)  # Assuming 'args' is a Namespace from argparse

    print('=============Begin training=============')
    if args.pretraining:
        pretraining.pretraining_loop(scheduler, model, loss_fn, regularization_fn, train_loader, test_loader, optim, logger, device, args, wandb, visual_train_loader, visual_test_loader)
        exit(0)
    if args.warmup:
        warmup = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lambda t: t / args.warmup)
    else:
        warmup = None
    pbar = tqdm(range(args.epoch))
    for epc in pbar:
        if args.warmup and epc < args.warmup:
            warmup.step()

        avg_loss, train_coef = train_fn(epc, model, loss_fn, regularization_fn, train_loader, optim, logger, device, args, wandb)
        if scheduler is not None and (args.lr_decay != 'cos' or epc >= args.warmup):
            scheduler.step()
        test_loss, test_coef = test_epoch(epc, model, test_loader, logger, device, args, wandb)
        if test_coef > best_coef:
            best_coef, best_epoch = test_coef, epc
            torch.save(model.state_dict(), './ckpt/' + args.model_name + '_best.pkl')

        pbar.set_description('Epoch: {}\tLoss: {:.4f}\tTrain Coef: {:.3f}\tTest Loss: {:.4f}\tTest Coef: {:.3f}'
              .format(epc, avg_loss, train_coef, test_loss, test_coef))
        if epc == args.epoch - 1:
            final_train_loss, final_train_coef, final_test_loss, final_test_coef = \
                avg_loss, train_coef, test_loss, test_coef
    torch.save(model.state_dict(), './ckpt/' + args.model_name + '_final.pkl')
    print('Best Test Coef: {:.3f}\tBest Test Epoch: {}'.format(best_coef, best_epoch))
    if not args.test:
        wandb.finish()
