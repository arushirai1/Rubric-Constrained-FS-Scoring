#CUDA_VISIBLE_DEVICES=1 python main.py --video-path /afs/cs.pitt.edu/usr0/arr159/figure_skating/CVPR22_GDLT/fis-v/swintx_avg_fps25_clip32 \
#--train-label-path /afs/cs.pitt.edu/usr0/arr159/figure_skating/CVPR22_GDLT/fis-v/train.txt \
#--test-label-path /afs/cs.pitt.edu/usr0/arr159/figure_skating/CVPR22_GDLT/fis-v/test.txt --score-type TES \
#--model-name 0207_clip_rubric_with_pooler_ortho_peak_restrict_first_quad --optim rmsprop  --ortho_loss  --peak_loss --clip-num 128 --lr 1e-5 --epoch 320 --n_encoder 1 --n_decoder 2 --n_query 7 \
#--alpha 1.0 --margin 1.0 --lr-decay cos --decay-rate 0.01 --dropout 0.3 --clip_embedding_path pooler_text_embeddings_and_weights.pth --n-workers 8 --restrict_first_quad &

#CUDA_VISIBLE_DEVICES=1 python main.py --video-path /afs/cs.pitt.edu/usr0/arr159/figure_skating/CVPR22_GDLT/fis-v/swintx_avg_fps25_clip32 \
#--train-label-path /afs/cs.pitt.edu/usr0/arr159/figure_skating/CVPR22_GDLT/fis-v/train.txt \
#--test-label-path /afs/cs.pitt.edu/usr0/arr159/figure_skating/CVPR22_GDLT/fis-v/test.txt --score-type TES \
#--model-name 0208_clip_rubric_with_pooler_ortho_peak_inject_clip_middle --optim rmsprop  --ortho_loss  --peak_loss --clip-num 128 --lr 1e-5 --epoch 320 --n_encoder 1 --n_decoder 2 --n_query 7 \
#--alpha 1.0 --margin 1.0 --lr-decay cos --decay-rate 0.01 --dropout 0.3 --clip_embedding_path pooler_text_embeddings_and_weights.pth --n-workers 8   --vision_vlm_inject --vision_vlm_inject_path /archive1/arr159/gdlt/clip/middle &


#CUDA_VISIBLE_DEVICES=3 python main.py --video-path /afs/cs.pitt.edu/usr0/arr159/figure_skating/CVPR22_GDLT/fis-v/swintx_avg_fps25_clip32 \
#--train-label-path /afs/cs.pitt.edu/usr0/arr159/figure_skating/CVPR22_GDLT/fis-v/train.txt \
#--test-label-path /afs/cs.pitt.edu/usr0/arr159/figure_skating/CVPR22_GDLT/fis-v/test.txt --score-type TES \
#--model-name 0207_clip_rubric_ortho_peak_restrict_first_quad_inject_clip_middle  --optim rmsprop  --ortho_loss  --peak_loss --clip-num 128 --lr 1e-5 --epoch 320 --n_encoder 1 --n_decoder 2 --n_query 7 \
#--alpha 1.0 --margin 1.0 --lr-decay cos --decay-rate 0.01 --dropout 0.3 --clip_embedding_path text_embeddings_and_weights.pth --n-workers 8 --restrict_first_quad \
# --vision_vlm_inject --vision_vlm_inject_path /archive1/arr159/gdlt/clip/middle &
#
#CUDA_VISIBLE_DEVICES=3 python main.py --video-path /afs/cs.pitt.edu/usr0/arr159/figure_skating/CVPR22_GDLT/fis-v/swintx_avg_fps25_clip32 \
#--train-label-path /afs/cs.pitt.edu/usr0/arr159/figure_skating/CVPR22_GDLT/fis-v/train.txt \
#--test-label-path /afs/cs.pitt.edu/usr0/arr159/figure_skating/CVPR22_GDLT/fis-v/test.txt --score-type TES \
#--model-name 0207_clip_rubric_ortho_peak_restrict_first_quad_action_rubric_mask --optim rmsprop --action_rubric_mask --ortho_loss  --peak_loss --clip-num 128 --lr 1e-5 --epoch 320 --n_encoder 1 --n_decoder 2 --n_query 7 \
#--alpha 1.0 --margin 1.0 --lr-decay cos --decay-rate 0.01 --dropout 0.3 --clip_embedding_path text_embeddings_and_weights.pth --n-workers 8 --restrict_first_quad &
#
#sleep 1200
#
#CUDA_VISIBLE_DEVICES=2 python main.py --video-path /afs/cs.pitt.edu/usr0/arr159/figure_skating/CVPR22_GDLT/fis-v/swintx_avg_fps25_clip32 \
#--train-label-path /afs/cs.pitt.edu/usr0/arr159/figure_skating/CVPR22_GDLT/fis-v/train.txt \
#--test-label-path /afs/cs.pitt.edu/usr0/arr159/figure_skating/CVPR22_GDLT/fis-v/test.txt --score-type TES \
#--model-name 0207_clip_rubric_ortho_peak_restrict_first_quad_inject_clip_middle_action_mask  --action_rubric_mask --optim rmsprop  --ortho_loss  --peak_loss --clip-num 128 --lr 1e-5 --epoch 320 --n_encoder 1 --n_decoder 2 --n_query 7 \
#--alpha 1.0 --margin 1.0 --lr-decay cos --decay-rate 0.01 --dropout 0.3 --clip_embedding_path text_embeddings_and_weights.pth --n-workers 8 --restrict_first_quad \
# --vision_vlm_inject --vision_vlm_inject_path /archive1/arr159/gdlt/clip/middle &
#
# CUDA_VISIBLE_DEVICES=2 python main.py --video-path /afs/cs.pitt.edu/usr0/arr159/figure_skating/CVPR22_GDLT/fis-v/swintx_avg_fps25_clip32 \
#--train-label-path /afs/cs.pitt.edu/usr0/arr159/figure_skating/CVPR22_GDLT/fis-v/train.txt \
#--test-label-path /afs/cs.pitt.edu/usr0/arr159/figure_skating/CVPR22_GDLT/fis-v/test.txt --score-type TES \
#--model-name 0207_clip_rubric_ortho_peak_restrict_first_quad_action_rubric_mask_clip_l1 --optim rmsprop --clip_l1_norm --ortho_loss  --peak_loss --clip-num 128 --lr 1e-5 --epoch 320 --n_encoder 1 --n_decoder 2 --n_query 7 \
#--alpha 1.0 --margin 1.0 --lr-decay cos --decay-rate 0.01 --dropout 0.3 --clip_embedding_path text_embeddings_and_weights.pth --n-workers 8 --restrict_first_quad &

CUDA_VISIBLE_DEVICES=3 python main.py --video-path /afs/cs.pitt.edu/usr0/arr159/figure_skating/CVPR22_GDLT/fis-v/swintx_avg_fps25_clip32 \
--train-label-path /afs/cs.pitt.edu/usr0/arr159/figure_skating/CVPR22_GDLT/fis-v/train.txt \
--test-label-path /afs/cs.pitt.edu/usr0/arr159/figure_skating/CVPR22_GDLT/fis-v/test.txt --score-type TES \
--model-name 0208_clip_rubric_ortho_peak_restrict_first_quad_action_rubric_mask_with_goe_sup --optim rmsprop --action_rubric_mask --ortho_loss  --peak_loss --clip-num 128 --lr 1e-5 --epoch 320 --n_encoder 1 --n_decoder 2 --n_query 7 \
--alpha 1.0 --margin 1.0 --lr-decay cos --decay-rate 0.01 --dropout 0.3 --clip_embedding_path text_embeddings_and_weights.pth --n-workers 8 --restrict_first_quad --predict_goe &
