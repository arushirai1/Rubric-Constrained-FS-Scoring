#CUDA_VISIBLE_GPUS=2 python main.py --video-path /afs/cs.pitt.edu/usr0/arr159/figure_skating/CVPR22_GDLT/fis-v/swintx_avg_fps25_clip32 --train-label-path /afs/cs.pitt.edu/usr0/arr159/figure_skating/CVPR22_GDLT/fis-v/train.txt --test-label-path /afs/cs.pitt.edu/usr0/arr159/figure_skating/CVPR22_GDLT/fis-v/test.txt --score-type TES --model-name 0110gdlt_bv_kb_q_5 --clip-num 128 --lr 0.01 --epoch 320 --n_encoder 1 --n_decoder 2 --n_query 5 --alpha 0.5 --margin 1.0 --lr-decay cos --decay-rate 0.01 --dropout 0.7 --n-workers 8 --gdlt &
#CUDA_VISIBLE_GPUS=2 python main.py --video-path /afs/cs.pitt.edu/usr0/arr159/figure_skating/CVPR22_GDLT/fis-v/swintx_avg_fps25_clip32 --train-label-path /afs/cs.pitt.edu/usr0/arr159/figure_skating/CVPR22_GDLT/fis-v/train.txt --test-label-path /afs/cs.pitt.edu/usr0/arr159/figure_skating/CVPR22_GDLT/fis-v/test.txt --score-type TES --model-name 0110gdlt_bv_kb_q_6 --clip-num 128 --lr 0.01 --epoch 320 --n_encoder 1 --n_decoder 2 --n_query 6 --alpha 0.5 --margin 1.0 --lr-decay cos --decay-rate 0.01 --dropout 0.7 --n-workers 8 --gdlt &
#CUDA_VISIBLE_GPUS=2 python main.py --video-path /afs/cs.pitt.edu/usr0/arr159/figure_skating/CVPR22_GDLT/fis-v/swintx_avg_fps25_clip32 --train-label-path /afs/cs.pitt.edu/usr0/arr159/figure_skating/CVPR22_GDLT/fis-v/train.txt --test-label-path /afs/cs.pitt.edu/usr0/arr159/figure_skating/CVPR22_GDLT/fis-v/test.txt --score-type TES --model-name 0110gdlt_bv_kb_q_7 --clip-num 128 --lr 0.01 --epoch 320 --n_encoder 1 --n_decoder 2 --n_query 7 --alpha 0.5 --margin 1.0 --lr-decay cos --decay-rate 0.01 --dropout 0.7 --n-workers 8 --gdlt &
#CUDA_VISIBLE_GPUS=3 python main.py --video-path /afs/cs.pitt.edu/usr0/arr159/figure_skating/CVPR22_GDLT/fis-v/swintx_avg_fps25_clip32 --train-label-path /afs/cs.pitt.edu/usr0/arr159/figure_skating/CVPR22_GDLT/fis-v/train.txt --test-label-path /afs/cs.pitt.edu/usr0/arr159/figure_skating/CVPR22_GDLT/fis-v/test.txt --score-type TES --model-name 0110gdlt_bv_kb_lr_1e-3 --clip-num 128 --lr 1e-3 --epoch 320 --n_encoder 1 --n_decoder 2 --n_query 4 --alpha 0.5 --margin 1.0 --lr-decay cos --decay-rate 0.01 --dropout 0.7 --n-workers 8 --gdlt &
#CUDA_VISIBLE_GPUS=3 python main.py --video-path /afs/cs.pitt.edu/usr0/arr159/figure_skating/CVPR22_GDLT/fis-v/swintx_avg_fps25_clip32 \
#--train-label-path /afs/cs.pitt.edu/usr0/arr159/figure_skating/CVPR22_GDLT/fis-v/train.txt \
#--test-label-path /afs/cs.pitt.edu/usr0/arr159/figure_skating/CVPR22_GDLT/fis-v/test.txt --score-type TES \
#--model-name 0110gdlt_bv_kb_d4 --clip-num 128 --lr 1e-2 --epoch 320 --n_encoder 1 --n_decoder 4 --n_query 4 \
#--alpha 0.5 --margin 1.0 --lr-decay cos --decay-rate 0.01 --dropout 0.7 --n-workers 8 --gdlt &

#CUDA_VISIBLE_DEVICES=3 python main.py --video-path /afs/cs.pitt.edu/usr0/arr159/figure_skating/CVPR22_GDLT/fis-v/swintx_avg_fps25_clip32 \
#--train-label-path /afs/cs.pitt.edu/usr0/arr159/figure_skating/CVPR22_GDLT/fis-v/train.txt \
#--test-label-path /afs/cs.pitt.edu/usr0/arr159/figure_skating/CVPR22_GDLT/fis-v/test.txt --score-type TES \
#--model-name 0122clip_inject_concat_middle --optim rmsprop --action_rubric_mask --peak_loss --ortho_loss --clip-num 128 --lr 1e-5 --epoch 320 --n_encoder 1 --n_decoder 2 --n_query 7 \
#--alpha 1.0 --margin 1.0 --lr-decay cos --decay-rate 0.01 --dropout 0.3 --clip_embedding_path text_embeddings_and_weights.pth --n-workers 8 --vision_vlm_inject --vision_vlm_inject_path /archive1/arr159/gdlt/clip/middle &
#
#CUDA_VISIBLE_DEVICES=3 python main.py --video-path /afs/cs.pitt.edu/usr0/arr159/figure_skating/CVPR22_GDLT/fis-v/swintx_avg_fps25_clip32 \
#--train-label-path /afs/cs.pitt.edu/usr0/arr159/figure_skating/CVPR22_GDLT/fis-v/train.txt \
#--test-label-path /afs/cs.pitt.edu/usr0/arr159/figure_skating/CVPR22_GDLT/fis-v/test.txt --score-type TES \
#--model-name 0122clip_inject_concat_middle_no_action_mask --optim rmsprop --peak_loss --ortho_loss --clip-num 128 --lr 1e-5 --epoch 320 --n_encoder 1 --n_decoder 2 --n_query 7 \
#--alpha 1.0 --margin 1.0 --lr-decay cos --decay-rate 0.01 --dropout 0.3 --clip_embedding_path text_embeddings_and_weights.pth --n-workers 8 --vision_vlm_inject --vision_vlm_inject_path /archive1/arr159/gdlt/clip/middle &
#
#
#CUDA_VISIBLE_DEVICES=3 python main.py --video-path /afs/cs.pitt.edu/usr0/arr159/figure_skating/CVPR22_GDLT/fis-v/swintx_avg_fps25_clip32 \
#--train-label-path /afs/cs.pitt.edu/usr0/arr159/figure_skating/CVPR22_GDLT/fis-v/train.txt \
#--test-label-path /afs/cs.pitt.edu/usr0/arr159/figure_skating/CVPR22_GDLT/fis-v/test.txt --score-type TES \
#--model-name 0122clip_inject_concat_start --optim rmsprop --action_rubric_mask --peak_loss --ortho_loss --clip-num 128 --lr 1e-5 --epoch 320 --n_encoder 1 --n_decoder 2 --n_query 7 \
#--alpha 1.0 --margin 1.0 --lr-decay cos --decay-rate 0.01 --dropout 0.3 --clip_embedding_path text_embeddings_and_weights.pth --n-workers 8 --vision_vlm_inject --vision_vlm_inject_path /archive1/arr159/gdlt/clip/start &
#CUDA_VISIBLE_DEVICES=2 python main.py --video-path /afs/cs.pitt.edu/usr0/arr159/figure_skating/CVPR22_GDLT/fis-v/swintx_avg_fps25_clip32 \
#--train-label-path /afs/cs.pitt.edu/usr0/arr159/figure_skating/CVPR22_GDLT/fis-v/train.txt \
#--test-label-path /afs/cs.pitt.edu/usr0/arr159/figure_skating/CVPR22_GDLT/fis-v/test.txt --score-type TES \
#--model-name 0122clip_inject_concat_end --optim rmsprop --action_rubric_mask --peak_loss --ortho_loss --clip-num 128 --lr 1e-5 --epoch 320 --n_encoder 1 --n_decoder 2 --n_query 7 \
#--alpha 1.0 --margin 1.0 --lr-decay cos --decay-rate 0.01 --dropout 0.3 --clip_embedding_path text_embeddings_and_weights.pth --n-workers 8 --vision_vlm_inject --vision_vlm_inject_path /archive1/arr159/gdlt/clip/end &
#CUDA_VISIBLE_DEVICES=2 python main.py --video-path /afs/cs.pitt.edu/usr0/arr159/figure_skating/CVPR22_GDLT/fis-v/swintx_avg_fps25_clip32 \
#--train-label-path /afs/cs.pitt.edu/usr0/arr159/figure_skating/CVPR22_GDLT/fis-v/train.txt \
#--test-label-path /afs/cs.pitt.edu/usr0/arr159/figure_skating/CVPR22_GDLT/fis-v/test.txt --score-type TES \
#--model-name 0122clip_inject_concat_random --optim rmsprop --action_rubric_mask --peak_loss --ortho_loss --clip-num 128 --lr 1e-5 --epoch 320 --n_encoder 1 --n_decoder 2 --n_query 7 \
#--alpha 1.0 --margin 1.0 --lr-decay cos --decay-rate 0.01 --dropout 0.3 --clip_embedding_path text_embeddings_and_weights.pth --n-workers 8 --vision_vlm_inject --vision_vlm_inject_path /archive1/arr159/gdlt/clip/random &
#
#CUDA_VISIBLE_DEVICES=2 python main.py --video-path /afs/cs.pitt.edu/usr0/arr159/figure_skating/CVPR22_GDLT/fis-v/swintx_avg_fps25_clip32 \
#--train-label-path /afs/cs.pitt.edu/usr0/arr159/figure_skating/CVPR22_GDLT/fis-v/train.txt \
#--test-label-path /afs/cs.pitt.edu/usr0/arr159/figure_skating/CVPR22_GDLT/fis-v/test.txt --score-type TES \
#--model-name 0122clip_inject_concat_random_baseline_sgd --optim sgd --peak_loss --ortho_loss --clip-num 128 --lr 1e-4 --epoch 320 --n_encoder 1 --n_decoder 2 --n_query 7 \
#--alpha 1.0 --margin 1.0 --hidden_dim 768 --lr-decay cos --decay-rate 0.01 --dropout 0.3 --n-workers 8 --vision_vlm_inject --vision_vlm_inject_path /archive1/arr159/gdlt/clip/random &
#
#CUDA_VISIBLE_DEVICES=2 python main.py --video-path /afs/cs.pitt.edu/usr0/arr159/figure_skating/CVPR22_GDLT/fis-v/swintx_avg_fps25_clip32 \
#--train-label-path /afs/cs.pitt.edu/usr0/arr159/figure_skating/CVPR22_GDLT/fis-v/train.txt \
#--test-label-path /afs/cs.pitt.edu/usr0/arr159/figure_skating/CVPR22_GDLT/fis-v/test.txt --score-type TES \
#--model-name 0122ours_baseline_sgd --optim sgd --peak_loss --ortho_loss --clip-num 128 --lr 1e-4 --epoch 320 --n_encoder 1 --n_decoder 2 --n_query 7 \
#--alpha 1.0 --margin 1.0 --hidden_dim 768 --lr-decay cos --decay-rate 0.01 --dropout 0.3 --n-workers 8  &
#
#
#CUDA_VISIBLE_DEVICES=1 python main.py --video-path /afs/cs.pitt.edu/usr0/arr159/figure_skating/CVPR22_GDLT/fis-v/swintx_avg_fps25_clip32 \
#--train-label-path /afs/cs.pitt.edu/usr0/arr159/figure_skating/CVPR22_GDLT/fis-v/train.txt \
#--test-label-path /afs/cs.pitt.edu/usr0/arr159/figure_skating/CVPR22_GDLT/fis-v/test.txt --score-type TES \
#--model-name 0122clip_inject_concat_random_gdlt --clip-num 128 --lr 1e-2 --epoch 320 --n_encoder 1 --n_decoder 2 --n_query 4 \
#--alpha 0.5 --margin 1.0 --hidden_dim 768 --lr-decay cos --decay-rate 0.01 --dropout 0.7 --n-workers 8 --gdlt --vision_vlm_inject --vision_vlm_inject_path /archive1/arr159/gdlt/clip/random &
#CUDA_VISIBLE_DEVICES=3 python main.py --video-path /afs/cs.pitt.edu/usr0/arr159/figure_skating/CVPR22_GDLT/fis-v/swintx_avg_fps25_clip32 --train-label-path /afs/cs.pitt.edu/usr0/arr159/figure_skating/CVPR22_GDLT/fis-v/train.txt --test-label-path /afs/cs.pitt.edu/usr0/arr159/figure_skating/CVPR22_GDLT/fis-v/test.txt --score-type TES --model-name 0122clip_inject_concat_start --optim rmsprop --action_rubric_mask --peak_loss --ortho_loss --clip-num 128 --lr 1e-5 --epoch 320 --n_encoder 1 --n_decoder 2 --n_query 7 --alpha 1.0 --margin 1.0 --lr-decay cos --decay-rate 0.01 --dropout 0.3 --clip_embedding_path text_embeddings_and_weights.pth --n-workers 8 --vision_vlm_inject --vision_vlm_inject_path /archive1/arr159/gdlt/clip/random --test --ckpt 0122clip_inject_concat_middle_no_action_mask_best
#CUDA_VISIBLE_DEVICES=3 python main.py --video-path /afs/cs.pitt.edu/usr0/arr159/figure_skating/CVPR22_GDLT/fis-v/swintx_avg_fps25_clip32 --train-label-path /afs/cs.pitt.edu/usr0/arr159/figure_skating/CVPR22_GDLT/fis-v/train.txt --test-label-path /afs/cs.pitt.edu/usr0/arr159/figure_skating/CVPR22_GDLT/fis-v/test.txt --score-type TES --model-name 0122clip_inject_concat_middle_no_action_mask_best --optim rmsprop --peak_loss --ortho_loss --clip-num 128 --lr 1e-5 --epoch 320 --n_encoder 1 --n_decoder 2 --n_query 7 --alpha 1.0 --margin 1.0 --lr-decay cos --decay-rate 0.01 --dropout 0.3 --clip_embedding_path text_embeddings_and_weights.pth --n-workers 8 --vision_vlm_inject --vision_vlm_inject_path /archive1/arr159/gdlt/clip/random --test --ckpt 0122clip_inject_concat_middle_no_action_mask_best
# ortho
CUDA_VISIBLE_DEVICES=4 python main.py --video-path /afs/cs.pitt.edu/usr0/arr159/figure_skating/CVPR22_GDLT/fis-v/swintx_avg_fps25_clip32 \
--train-label-path /afs/cs.pitt.edu/usr0/arr159/figure_skating/CVPR22_GDLT/fis-v/train.txt \
--test-label-path /afs/cs.pitt.edu/usr0/arr159/figure_skating/CVPR22_GDLT/fis-v/test.txt --score-type TES \
--model-name 0124_pain_1e2 --optim rmsprop  --clip-num 128 --lr 1e-5 --epoch 320 --n_encoder 1 --n_decoder 2 --n_query 7 \
--alpha 1.0 --margin 1.0 --lr-decay cos --decay-rate 0.01 --dropout 0.3 --clip_embedding_path text_embeddings_and_weights.pth --n-workers 8 &

CUDA_VISIBLE_DEVICES=4 python main.py --video-path /afs/cs.pitt.edu/usr0/arr159/figure_skating/CVPR22_GDLT/fis-v/swintx_avg_fps25_clip32 \
--train-label-path /afs/cs.pitt.edu/usr0/arr159/figure_skating/CVPR22_GDLT/fis-v/train.txt \
--test-label-path /afs/cs.pitt.edu/usr0/arr159/figure_skating/CVPR22_GDLT/fis-v/test.txt --score-type TES \
--model-name 0124_plain_1e8 --optim rmsprop  --clip-num 128 --lr 1e-5 --epoch 320 --n_encoder 1 --n_decoder 8 --n_query 7 \
--alpha 1.0 --margin 1.0 --lr-decay cos --decay-rate 0.01 --dropout 0.3 --clip_embedding_path text_embeddings_and_weights.pth --n-workers 8 &

# ortho + action masking
CUDA_VISIBLE_DEVICES=5 python main.py --video-path /afs/cs.pitt.edu/usr0/arr159/figure_skating/CVPR22_GDLT/fis-v/swintx_avg_fps25_clip32 \
--train-label-path /afs/cs.pitt.edu/usr0/arr159/figure_skating/CVPR22_GDLT/fis-v/train.txt \
--test-label-path /afs/cs.pitt.edu/usr0/arr159/figure_skating/CVPR22_GDLT/fis-v/test.txt --score-type TES \
--model-name 0124_ortho_action_mask_1e2 --optim rmsprop  --ortho_loss --clip-num 128 --lr 1e-5 --epoch 320 --n_encoder 1 --n_decoder 2 --n_query 7 \
--alpha 1.0 --margin 1.0 --lr-decay cos --decay-rate 0.01 --dropout 0.3 --action_rubric_mask --clip_embedding_path text_embeddings_and_weights.pth --n-workers 8 &

CUDA_VISIBLE_DEVICES=5 python main.py --video-path /afs/cs.pitt.edu/usr0/arr159/figure_skating/CVPR22_GDLT/fis-v/swintx_avg_fps25_clip32 \
--train-label-path /afs/cs.pitt.edu/usr0/arr159/figure_skating/CVPR22_GDLT/fis-v/train.txt \
--test-label-path /afs/cs.pitt.edu/usr0/arr159/figure_skating/CVPR22_GDLT/fis-v/test.txt --score-type TES \
--model-name 0124_ortho_action_mask_1e8 --optim rmsprop  --ortho_loss --clip-num 128 --lr 1e-5 --epoch 320 --n_encoder 1 --n_decoder 8 --n_query 7 \
--alpha 1.0 --margin 1.0 --lr-decay cos --decay-rate 0.01 --dropout 0.3 --action_rubric_mask --clip_embedding_path text_embeddings_and_weights.pth --n-workers 8 &

sleep 1200
#--action_rubric_mask --peak_loss
# ortho peak
CUDA_VISIBLE_DEVICES=4 python main.py --video-path /afs/cs.pitt.edu/usr0/arr159/figure_skating/CVPR22_GDLT/fis-v/swintx_avg_fps25_clip32 \
--train-label-path /afs/cs.pitt.edu/usr0/arr159/figure_skating/CVPR22_GDLT/fis-v/train.txt \
--test-label-path /afs/cs.pitt.edu/usr0/arr159/figure_skating/CVPR22_GDLT/fis-v/test.txt --score-type TES \
--model-name 0124_ortho_peak_1e2 --optim rmsprop  --ortho_loss --peak_loss --clip-num 128 --lr 1e-5 --epoch 320 --n_encoder 1 --n_decoder 2 --n_query 7 \
--alpha 1.0 --margin 1.0 --lr-decay cos --decay-rate 0.01 --dropout 0.3 --clip_embedding_path text_embeddings_and_weights.pth --n-workers 8 &

CUDA_VISIBLE_DEVICES=4 python main.py --video-path /afs/cs.pitt.edu/usr0/arr159/figure_skating/CVPR22_GDLT/fis-v/swintx_avg_fps25_clip32 \
--train-label-path /afs/cs.pitt.edu/usr0/arr159/figure_skating/CVPR22_GDLT/fis-v/train.txt \
--test-label-path /afs/cs.pitt.edu/usr0/arr159/figure_skating/CVPR22_GDLT/fis-v/test.txt --score-type TES \
--model-name 0124_ortho_peak_1e8 --optim rmsprop  --ortho_loss  --peak_loss --clip-num 128 --lr 1e-5 --epoch 320 --n_encoder 1 --n_decoder 8 --n_query 7 \
--alpha 1.0 --margin 1.0 --lr-decay cos --decay-rate 0.01 --dropout 0.3 --clip_embedding_path text_embeddings_and_weights.pth --n-workers 8 &

# ortho peak l1 -> 1e2d

CUDA_VISIBLE_DEVICES=5 python main.py --video-path /afs/cs.pitt.edu/usr0/arr159/figure_skating/CVPR22_GDLT/fis-v/swintx_avg_fps25_clip32 \
--train-label-path /afs/cs.pitt.edu/usr0/arr159/figure_skating/CVPR22_GDLT/fis-v/train.txt \
--test-label-path /afs/cs.pitt.edu/usr0/arr159/figure_skating/CVPR22_GDLT/fis-v/test.txt --score-type TES \
--model-name 0124_ortho_peak_l1_1e2 --optim rmsprop  --clip_l1_norm --ortho_loss --peak_loss --clip-num 128 --lr 1e-5 --epoch 320 --n_encoder 1 --n_decoder 2 --n_query 7 \
--alpha 1.0 --margin 1.0 --lr-decay cos --decay-rate 0.01 --dropout 0.3 --clip_embedding_path text_embeddings_and_weights.pth --n-workers 8 &

CUDA_VISIBLE_DEVICES=5 python main.py --video-path /afs/cs.pitt.edu/usr0/arr159/figure_skating/CVPR22_GDLT/fis-v/swintx_avg_fps25_clip32 \
--train-label-path /afs/cs.pitt.edu/usr0/arr159/figure_skating/CVPR22_GDLT/fis-v/train.txt \
--test-label-path /afs/cs.pitt.edu/usr0/arr159/figure_skating/CVPR22_GDLT/fis-v/test.txt --score-type TES \
--model-name 0124_ortho_peak_l1_1e8 --optim rmsprop --clip_l1_norm  --ortho_loss  --peak_loss --clip-num 128 --lr 1e-5 --epoch 320 --n_encoder 1 --n_decoder 8 --n_query 7 \
--alpha 1.0 --margin 1.0 --lr-decay cos --decay-rate 0.01 --dropout 0.3 --clip_embedding_path text_embeddings_and_weights.pth --n-workers 8 &
