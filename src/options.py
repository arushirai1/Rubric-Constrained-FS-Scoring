import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--video-path', type=str, default='../action_assessment/rg_feat/swintx_avg_fps25_clip32')
parser.add_argument('--clip-num', type=int, default=68)
parser.add_argument('--n-workers', type=int, default=8)

parser.add_argument('--train-label-path', type=str, default='../action_assessment/rg_feat/train.txt')
parser.add_argument('--test-label-path', type=str, default='../action_assessment/rg_feat/test.txt')

parser.add_argument('--dataset-name', type=str, default='fisv')
parser.add_argument('--action-type', type=str, default='Ball')
parser.add_argument('--score-type', type=str, default='Total_Score')
parser.add_argument('--clip_embedding_path', type=str, default=None)

parser.add_argument('--model-name', type=str, default='test', help='name used to save model and logs')
parser.add_argument("--ckpt", default=None, help="ckpt for pretrained model")
parser.add_argument("--test", action='store_true', help="only evaluate, don't train")

parser.add_argument('--epoch', type=int, default=400)
parser.add_argument('--batch', type=int, default=32)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight-decay', type=float, default=1e-4)

parser.add_argument('--optim', type=str, default='sgd')

parser.add_argument("--lr-decay", type=str, default=None, help='use what decay scheduler')
parser.add_argument("--decay-rate", type=float, default=0.1, help="lr decay rate")
parser.add_argument("--warmup", type=int, default=0, help="warmup epoch")

parser.add_argument('--in_dim', type=int, default=1024)
parser.add_argument('--hidden_dim', type=int, default=256)
parser.add_argument('--n_head', type=int, default=1)
parser.add_argument('--n_encoder', type=int, default=1)
parser.add_argument('--n_decoder', type=int, default=1)
parser.add_argument('--n_query', type=int, default=1)

parser.add_argument('--alpha', type=float, default=0.0)
parser.add_argument('--margin', type=float, default=0.0)

parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--rescaling', action='store_true')
parser.add_argument('--ortho_loss', action='store_true')
parser.add_argument('--peak_loss', action='store_true')
parser.add_argument('--sliding_peak_loss', action='store_true')
parser.add_argument('--predict_bv', action='store_true')
parser.add_argument('--predict_goe', action='store_true')
parser.add_argument('--predict_deductions', action='store_true')
parser.add_argument('--l2_regularization', action='store_true')
parser.add_argument('--clip_l1_norm', action='store_true')
parser.add_argument('--clip_sigmoid', action='store_true')
parser.add_argument('--clip_relu', action='store_true')
parser.add_argument('--action_rubric_mask', action='store_true')
parser.add_argument('--gdlt', action='store_true')
parser.add_argument('--grad_stats', action='store_true')
parser.add_argument('--drop_last', action='store_true')
parser.add_argument('--extrema_penalty', action='store_true')
parser.add_argument('--restrict_first_quad', action='store_true')
parser.add_argument('--pretraining', action='store_true')
parser.add_argument('--randomly_sample_dissim', action='store_true')
parser.add_argument('--enforce_dissimilarity', action='store_true')
parser.add_argument('--curriculum', action='store_true')
parser.add_argument('--use_visual_text_triplet_loss', action='store_true')
parser.add_argument('--use_visual_only_pretraining_loss', action='store_true')
parser.add_argument('--joint_pretraining', action='store_true')
parser.add_argument('--reg_weight', type=float, default=1)

parser.add_argument('--rubric_simplified', action='store_true', default=False)
parser.add_argument('--rubric_use_text_prompt', action='store_true', default=False)
parser.add_argument('--rubric_hand_negatives', action='store_true', default=False)
parser.add_argument('--vis_pos_only', action='store_true')
parser.add_argument('--vis_neg_only', action='store_true')
parser.add_argument('--visual_triplet_loss', action='store_true')
parser.add_argument('--dist_metric', default='cosine')
parser.add_argument('--visual_triplet_margin', type=float, default=0.5)
parser.add_argument('--pretraining_threshold_pos', type=float, default=2)
parser.add_argument('--pretraining_threshold_neg', type=float, default=-2)
parser.add_argument('--pretraining_agg', type=str, default='default')

parser.add_argument('--vision_vlm_inject', action='store_true')
parser.add_argument('--vision_vlm_inject_path', type=str, default=None)
parser.add_argument('--fisvtofs800_pretrain', action='store_true', default=False)


parser.add_argument('--vid_id_to_element_list_path', type=str, default='elements_preprocessing/vid_id_to_element_list.json') # change this to absolute path if needed
