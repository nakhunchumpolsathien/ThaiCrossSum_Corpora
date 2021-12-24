The code is based on https://github.com/ZNLP/NCLS-Corpora.

*word-level-supervised*: supervised pre-training with word-level knowledge distillation.

*sent-level-supervised*: supervised pre-training with sentence-level knowledge distillation.

*rl-sim*: reinforcement learning fine-tuning with cross-lingual similarity and ROUGE scores as rewards.

### Training:

python train.py -config train.json

### Translating:

python translate.py -config translate.json
