# relu
CUDA_VISIBLE_DEVICES=3 python main.py --video-path /afs/cs.pitt.edu/usr0/arr159/figure_skating/CVPR22_GDLT/fis-v/swintx_avg_fps25_clip32 \
--train-label-path /afs/cs.pitt.edu/usr0/arr159/figure_skating/CVPR22_GDLT/fis-v/train.txt \
--test-label-path /afs/cs.pitt.edu/usr0/arr159/figure_skating/CVPR22_GDLT/fis-v/test.txt --score-type TES \
--model-name 0125relu_clip_rubric_NN --optim rmsprop --peak_loss --ortho_loss --clip-num 128 --lr 1e-5 --epoch 10 --n_encoder 1 --n_decoder 2 --n_query 7 \
--alpha 1.0 --margin 1.0 --hidden_dim 768 --lr-decay cos --decay-rate 0.01 --dropout 0.3 \
--clip_embedding_path text_embeddings_and_weights.pth --n-workers 8 --grad_stats --clip_relu &

# sigmoid
CUDA_VISIBLE_DEVICES=3 python main.py --video-path /afs/cs.pitt.edu/usr0/arr159/figure_skating/CVPR22_GDLT/fis-v/swintx_avg_fps25_clip32 \
--train-label-path /afs/cs.pitt.edu/usr0/arr159/figure_skating/CVPR22_GDLT/fis-v/train.txt \
--test-label-path /afs/cs.pitt.edu/usr0/arr159/figure_skating/CVPR22_GDLT/fis-v/test.txt --score-type TES \
--model-name 0125sigmoid_clip_rubric_NN --optim rmsprop --peak_loss --ortho_loss --clip-num 128 --lr 1e-5 --epoch 10 --n_encoder 1 --n_decoder 2 --n_query 7 \
--alpha 1.0 --margin 1.0 --hidden_dim 768 --lr-decay cos --decay-rate 0.01 --dropout 0.3 \
--clip_embedding_path text_embeddings_and_weights.pth --n-workers 8 --grad_stats  --clip_sigmoid &

# SGD for clip rubric

CUDA_VISIBLE_DEVICES=1 python main.py --video-path /afs/cs.pitt.edu/usr0/arr159/figure_skating/CVPR22_GDLT/fis-v/swintx_avg_fps25_clip32 \
--train-label-path /afs/cs.pitt.edu/usr0/arr159/figure_skating/CVPR22_GDLT/fis-v/train.txt \
--test-label-path /afs/cs.pitt.edu/usr0/arr159/figure_skating/CVPR22_GDLT/fis-v/test.txt --score-type TES \
--model-name 0125ours_baseline_sgd_clip_rubric_NN --optim sgd --peak_loss --ortho_loss --clip-num 128 --lr 1e-5 --epoch 10 --n_encoder 1 --n_decoder 2 --n_query 7 \
--alpha 1.0 --margin 1.0 --hidden_dim 768 --lr-decay cos --decay-rate 0.01 --dropout 0.3 \
--clip_embedding_path text_embeddings_and_weights.pth --n-workers 8 --grad_stats &

#sleep 120

# RMSProp for clip rubric

CUDA_VISIBLE_DEVICES=1 python main.py --video-path /afs/cs.pitt.edu/usr0/arr159/figure_skating/CVPR22_GDLT/fis-v/swintx_avg_fps25_clip32 \
--train-label-path /afs/cs.pitt.edu/usr0/arr159/figure_skating/CVPR22_GDLT/fis-v/train.txt \
--test-label-path /afs/cs.pitt.edu/usr0/arr159/figure_skating/CVPR22_GDLT/fis-v/test.txt --score-type TES \
--model-name 0125ours_baseline_rmsprop_clip_rubric_NN --optim rmsprop --peak_loss --ortho_loss --clip-num 128 --lr 1e-5 --epoch 10 --n_encoder 1 --n_decoder 2 --n_query 7 \
--alpha 1.0 --margin 1.0 --hidden_dim 768 --lr-decay cos --decay-rate 0.01 --dropout 0.3 \
--clip_embedding_path text_embeddings_and_weights.pth --n-workers 8 --grad_stats &

# RMSProp w/ lr rate for clip rubric

CUDA_VISIBLE_DEVICES=2 python main.py --video-path /afs/cs.pitt.edu/usr0/arr159/figure_skating/CVPR22_GDLT/fis-v/swintx_avg_fps25_clip32 \
--train-label-path /afs/cs.pitt.edu/usr0/arr159/figure_skating/CVPR22_GDLT/fis-v/train.txt \
--test-label-path /afs/cs.pitt.edu/usr0/arr159/figure_skating/CVPR22_GDLT/fis-v/test.txt --score-type TES \
--model-name 0125ours_baseline_rmsprop_clip_rubric_NN_with_lr1e-4 --optim rmsprop --peak_loss --ortho_loss --clip-num 128 --lr 1e-4 --epoch 10 --n_encoder 1 --n_decoder 2 --n_query 7 \
--alpha 1.0 --margin 1.0 --hidden_dim 768 --lr-decay cos --decay-rate 0.01 --dropout 0.3 \
--clip_embedding_path text_embeddings_and_weights.pth --n-workers 8 --grad_stats &

# SGD for no clip rubric
CUDA_VISIBLE_DEVICES=2 python main.py --video-path /afs/cs.pitt.edu/usr0/arr159/figure_skating/CVPR22_GDLT/fis-v/swintx_avg_fps25_clip32 \
--train-label-path /afs/cs.pitt.edu/usr0/arr159/figure_skating/CVPR22_GDLT/fis-v/train.txt \
--test-label-path /afs/cs.pitt.edu/usr0/arr159/figure_skating/CVPR22_GDLT/fis-v/test.txt --score-type TES \
--model-name 0125ours_baseline_sgd_NN --optim sgd --peak_loss --ortho_loss --clip-num 128 --lr 1e-4 --epoch 10 --n_encoder 1 --n_decoder 2 --n_query 7 \
--alpha 1.0 --margin 1.0 --hidden_dim 768 --lr-decay cos --decay-rate 0.01 --dropout 0.3 --n-workers 8 --grad_stats &
#
# RMSProp for no clip rubric
CUDA_VISIBLE_DEVICES=2 python main.py --video-path /afs/cs.pitt.edu/usr0/arr159/figure_skating/CVPR22_GDLT/fis-v/swintx_avg_fps25_clip32 \
--train-label-path /afs/cs.pitt.edu/usr0/arr159/figure_skating/CVPR22_GDLT/fis-v/train.txt \
--test-label-path /afs/cs.pitt.edu/usr0/arr159/figure_skating/CVPR22_GDLT/fis-v/test.txt --score-type TES \
--model-name 0125ours_baseline_rmsprop_NN --optim rmsprop --peak_loss --ortho_loss --clip-num 128 --lr 1e-4 --epoch 10 --n_encoder 1 --n_decoder 2 --n_query 7 \
--alpha 1.0 --margin 1.0 --hidden_dim 768 --lr-decay cos --decay-rate 0.01 --dropout 0.3 --n-workers 8 --grad_stats &