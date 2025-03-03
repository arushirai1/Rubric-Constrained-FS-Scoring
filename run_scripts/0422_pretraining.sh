#0304clip_visual_triplet_margin_relative_margin
#0321clip_simplified_regularization_pretraining_best

## visual first, visual text pretraining lr 1e-2
#CUDA_VISIBLE_DEVICES=7 python main.py --video-path /afs/cs.pitt.edu/usr0/arr159/figure_skating/CVPR22_GDLT/fis-v/swintx_avg_fps25_clip32 --train-label-path /afs/cs.pitt.edu/usr0/arr159/figure_skating/CVPR22_GDLT/fis-v/train.txt \
#--test-label-path /afs/cs.pitt.edu/usr0/arr159/figure_skating/CVPR22_GDLT/fis-v/test.txt --score-type TES \
#--model-name 0422clip_simplified_regularization_w_visual_triplet_margin_relative_margin_pretraining_lr1e-2 --pretraining_threshold_pos 2 --pretraining_threshold_neg -2 --pretraining --pretraining_agg embed  --clip_embedding_path text_embeddings_and_weights.pth --clip-num 128 --lr 1e-2 \
#--epoch 300 --n_encoder 1 --n_decoder 2 --n_query 7 --batch 32 --margin 0.5 --rubric_simplified --peak_loss --ortho_loss --ckpt 0304clip_visual_triplet_margin_relative_margin_pretraining_best &
#
## visual first, visual text pretraining lr 1e-5
#CUDA_VISIBLE_DEVICES=7 python main.py --video-path /afs/cs.pitt.edu/usr0/arr159/figure_skating/CVPR22_GDLT/fis-v/swintx_avg_fps25_clip32 --train-label-path /afs/cs.pitt.edu/usr0/arr159/figure_skating/CVPR22_GDLT/fis-v/train.txt \
#--test-label-path /afs/cs.pitt.edu/usr0/arr159/figure_skating/CVPR22_GDLT/fis-v/test.txt --score-type TES \
#--model-name 0422clip_simplified_regularization_w_visual_triplet_margin_relative_margin_pretraining_lr1e-5 --pretraining_threshold_pos 2 --pretraining_threshold_neg -2 --pretraining --pretraining_agg embed  --clip_embedding_path text_embeddings_and_weights.pth --clip-num 128 --lr 1e-5 \
#--epoch 300 --n_encoder 1 --n_decoder 2 --n_query 7 --batch 32 --margin 0.5 --rubric_simplified --peak_loss --ortho_loss --ckpt 0304clip_visual_triplet_margin_relative_margin_pretraining_best &

# visual text first, visual pretraining lr 1e-2
CUDA_VISIBLE_DEVICES=7 python main.py --video-path /afs/cs.pitt.edu/usr0/arr159/figure_skating/CVPR22_GDLT/fis-v/swintx_avg_fps25_clip32 --train-label-path /afs/cs.pitt.edu/usr0/arr159/figure_skating/CVPR22_GDLT/fis-v/train.txt --test-label-path /afs/cs.pitt.edu/usr0/arr159/figure_skating/CVPR22_GDLT/fis-v/test.txt --score-type TES \
--model-name 0422clip_visual_triplet_margin_relative_margin_w_0321clip_visual_simplified_text_reg_pretraininglr1e-2 --pretraining_threshold_pos 0 --pretraining_threshold_neg 0 --pretraining --pretraining_agg embed --clip_embedding_path text_embeddings_and_weights.pth --clip-num 128 \
--lr 1e-2 --epoch 300 --n_encoder 1 --n_decoder 2 --n_query 7 --batch 32 --visual_triplet_loss --visual_triplet_margin -1 --ckpt 0321clip_simplified_regularization_pretraining_best --use_visual_only_pretraining_loss &

# visual text first, visual pretraining lr 1e-5
CUDA_VISIBLE_DEVICES=7 python main.py --video-path /afs/cs.pitt.edu/usr0/arr159/figure_skating/CVPR22_GDLT/fis-v/swintx_avg_fps25_clip32 --train-label-path /afs/cs.pitt.edu/usr0/arr159/figure_skating/CVPR22_GDLT/fis-v/train.txt --test-label-path /afs/cs.pitt.edu/usr0/arr159/figure_skating/CVPR22_GDLT/fis-v/test.txt --score-type TES \
--model-name 0422clip_visual_triplet_margin_relative_margin_w_0321clip_visual_simplified_text_reg_pretraininglr1e-5 --pretraining_threshold_pos 0 --pretraining_threshold_neg 0 --pretraining --pretraining_agg embed --clip_embedding_path text_embeddings_and_weights.pth --clip-num 128 \
--lr 1e-5 --epoch 300 --n_encoder 1 --n_decoder 2 --n_query 7 --batch 32 --visual_triplet_loss --visual_triplet_margin -1 --ckpt 0321clip_simplified_regularization_pretraining_best --use_visual_only_pretraining_loss &

# visual text first, visual pretraining lr 1e-2
CUDA_VISIBLE_DEVICES=0 python main.py --video-path /afs/cs.pitt.edu/usr0/arr159/figure_skating/CVPR22_GDLT/fis-v/swintx_avg_fps25_clip32 --train-label-path /afs/cs.pitt.edu/usr0/arr159/figure_skating/CVPR22_GDLT/fis-v/train.txt --test-label-path /afs/cs.pitt.edu/usr0/arr159/figure_skating/CVPR22_GDLT/fis-v/test.txt --score-type TES \
--model-name 422clip_visual_triplet_margin_0.05_w_simp_hand_neg_triplet_vnt_pretraining_lr1e-2 --pretraining_threshold_pos 0 --pretraining_threshold_neg 0 --pretraining --pretraining_agg embed  --clip_embedding_path text_embeddings_and_weights.pth --clip-num 128 \
--lr 1e-2 --epoch 300 --n_encoder 1 --n_decoder 2 --n_query 7 --batch 32 --visual_triplet_loss --visual_triplet_margin 0.05 \
--ckpt 0417clip_simplified_hand_neg_triplet_vnt_loss_reg_weight_0.1_margin_0.5_thresh1.3_pretraining_best --use_visual_only_pretraining_loss &

# visual text first, visual pretraining lr 1e-5
CUDA_VISIBLE_DEVICES=0 python main.py --video-path /afs/cs.pitt.edu/usr0/arr159/figure_skating/CVPR22_GDLT/fis-v/swintx_avg_fps25_clip32 --train-label-path /afs/cs.pitt.edu/usr0/arr159/figure_skating/CVPR22_GDLT/fis-v/train.txt --test-label-path /afs/cs.pitt.edu/usr0/arr159/figure_skating/CVPR22_GDLT/fis-v/test.txt --score-type TES \
--model-name 422clip_visual_triplet_margin_0.05_w_simp_hand_neg_triplet_vnt_pretraining_lr1e-5 --pretraining_threshold_pos 0 --pretraining_threshold_neg 0 --pretraining --pretraining_agg embed  --clip_embedding_path text_embeddings_and_weights.pth --clip-num 128 \
--lr 1e-5 --epoch 300 --n_encoder 1 --n_decoder 2 --n_query 7 --batch 32 --visual_triplet_loss --visual_triplet_margin 0.05 \
--ckpt 0417clip_simplified_hand_neg_triplet_vnt_loss_reg_weight_0.1_margin_0.5_thresh1.3_pretraining_best --use_visual_only_pretraining_loss &
