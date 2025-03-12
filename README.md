# Rubric-Constrained Figure Skating Scoring

[![Paper](https://img.shields.io/badge/Paper-PDF-red)](https://openaccess.thecvf.com/content/WACV2025/papers/Rai_Rubric-Constrained_Figure_Skating_Scoring_WACV_2025_paper.pdf)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Development Notice**: Ready for use!

This repository contains the implementation of [WACV 2025](https://openaccess.thecvf.com/content/WACV2025/papers/Rai_Rubric-Constrained_Figure_Skating_Scoring_WACV_2025_paper.pdf) paper of an interpretable-by-design approach for figure skating quality assessment which leverages rubric information used by judges and performs implicit segmentation of elements without additional segment annotations. This work builds upon the [codebase](https://github.com/xuangch/CVPR22_GDLT) of CVPR2022 paper "Likert Scoring with Grade Decoupling for Long-term Action Assessment".

## Overview

We present a novel approach for figure skating scoring that addresses the limitations of existing methods in action quality assessment. Our method introduces three key features:

1. **Implicit Element Segmentation**: We use learnable element queries in a transformer to automatically segment figure skating elements without requiring additional annotations. This allows for element-level analysis of performance quality.

2. **Rubric-Constrained Scoring**: We leverage natural language rubrics used by judges to score each element, making the scoring process more interpretable and aligned with expert judgment.

3. **Difficulty Isolation**: By using the element list (routine sequence) to isolate difficulty, we focus on assessing how well elements are executed, similar to how judges evaluate performances.

These components work together to improve both the interpretability of the scoring process and the accuracy of the results, making our approach more practical for athletes and coaches while maintaining high scoring precision and ranking capability.

## Environment Setup

- NVIDIA GTX 1080 GPU
- CUDA: 10.2
- Python: 3.9.7
- PyTorch: 1.10.1+cu102

## Installation

1. Clone the repository:
```bash
git clone https://github.com/arushirai1/Rubric-Constrained-FS-Scoring.git
cd Rubric-Constrained-FS-Scoring/src
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset

The features and label files for the Fis-V dataset can be downloaded [here](https://1drv.ms/u/s!AqXkt0Mw7p9llWEihc533CB87U5P?e=EadhCo). I store this in the data folder.

The FS800 dataset is also supported in this implementation. Please visit this repo for instructions on how to download, note this is a subset of FS1000 (https://github.com/AndyFrancesco29/Audio-Visual-Figure-Skating).

## Pretrained weights
You can download the best pretrained weights [here](https://www.dropbox.com/scl/fo/g6w1c270oqomgdwpw1nt6/ACcvXW7HxcnoTSzeE0gUcxQ?rlkey=hjtf85wtszuqh6g0u9yys7mjz&st=s0dgbxgs&dl=0).

## Usage

### Pretraining

First, you need to run the pretraining phase which jointly learns visual-text alignment and visual-rank aware features:

```bash
python main.py \
    --video-path ../data/fis-v/swintx_avg_fps25_clip32 \
    --train-label-path ../data/fis-v/train.txt \
    --test-label-path ../data/fis-v/test.txt \
    --score-type TES \
    --model-name your_model_name \
    --pretraining_threshold_pos 1.3 \
    --pretraining_threshold_neg -2 \
    --pretraining \
    --pretraining_agg embed \
    --clip_embedding_path text_embeddings_and_weights.pth \
    --clip-num 128 \
    --lr 1e-2 \
    --epoch 300 \
    --n_encoder 1 \
    --n_decoder 2 \
    --n_query 7 \
    --batch 32 \
    --use_visual_text_triplet_loss \
    --rubric_hand_negatives \
    --rubric_simplified \
    --peak_loss \
    --ortho_loss \
    --dist_metric cosine \
    --margin 0.5 \
    --reg_weight 0.1 \
    --visual_triplet_loss \
    --visual_triplet_margin 0.5 \
    --joint_pretraining \
    --vid_id_to_element_list_path ./elements_preprocessing/vid_id_to_element_list.json
```

### Training

After pretraining, you can run the main training phase, see the pretraining checkpoint:

```bash
python main.py \
    --video-path ../data/fis-v/swintx_avg_fps25_clip32 \
    --train-label-path ../data/fis-v/train.txt \
    --test-label-path ../data/fis-v/test.txt \
    --score-type TES \
    --model-name your_model_name \
    --action_rubric_mask \
    --ortho_loss \
    --peak_loss \
    --optim rmsprop \
    --clip-num 128 \
    --lr 1e-5 \
    --epoch 320 \
    --n_encoder 1 \
    --n_decoder 2 \
    --n_query 7 \
    --lr-decay cos \
    --decay-rate 0.01 \
    --dropout 0.3 \
    --vid_id_to_element_list_path elements_preprocessing/vid_id_to_element_list.json \
    --clip_embedding_path text_embeddings_and_weights.pth \
    --n-workers 4 \
    --rescaling \
    --rubric_simplified \
    --ckpt ../data/ckpt/0903_joint_pretraining_vis_triplet_margin_0.5_pretraining_best
```

### Testing

```bash
python main.py \
    --video-path ../data/fis-v/swintx_avg_fps25_clip32 \
    --train-label-path ../data/fis-v/train.txt \
    --test-label-path ../data/fis-v/test.txt \
    --score-type TES \
    --action_rubric_mask \
    --clip-num 128 \
    --n_encoder 1 \
    --n_decoder 2 \
    --n_query 7 \
    --clip_embedding_path elements_preprocessing/text_embeddings_and_weights.pth \
    --n-workers 4 \
    --rescaling \
    --rubric_simplified \
    --vid_id_to_element_list_path elements_preprocessing/vid_id_to_element_list.json \
    --test \
    --ckpt ../data/0903_joint_pretraining_vis_triplet_margin_0.5_action_masking_finetune_best

```

## Baseline Replication

There are some aspects that only the GDLT approach uses in the following files:

- `models/triplet_loss.py`: Contains the GDLT triplet loss implementation
- `models/loss.py`: Contains the combined loss function using GDLT losses

This baseline is used for comparison with our method. These files are not used in our implementation.


## Citation

If you use this code in your research, please cite our paper:

```
@InProceedings{Rai_2025_WACV,
    author    = {Rai, Arushi and Kovashka, Adriana},
    title     = {Rubric-Constrained Figure Skating Scoring},
    booktitle = {Proceedings of the Winter Conference on Applications of Computer Vision (WACV)},
    month     = {February},
    year      = {2025},
    pages     = {9087-9095}
}
```

## Acknowledgments

This work builds upon the CVPR2022 paper "Likert Scoring with Grade Decoupling for Long-term Action Assessment" and their codebase. We thank the original authors for their valuable contributions.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2024 Arushi Rai

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```


## Contact

Feel free to email me at arr159 [at] pitt [dot] edu for questions and future collaborations.

