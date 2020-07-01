# IMAP
This repository includes the implementation of IMAP model.

Requirements: Torch, Python 3.6 and PyTorch 0.4.1.
## Data Preparation
### Download Dataset
Download the [VisualGenome dataset](http://visualgenome.org/api/v0/api_home.html), the [captions](http://visualgenome.org/static/data/dataset/paragraphs_v1.json.zip), and the [training](https://cs.stanford.edu/people/ranjaykrishna/im2p/train_split.json), [val](https://cs.stanford.edu/people/ranjaykrishna/im2p/val_split.json) and [test](https://cs.stanford.edu/people/ranjaykrishna/im2p/test_split.json) splits json files. Then, install the [Torch](http://torch.ch/) environment following [DenseCap](https://github.com/jcjohnson/densecap) step by step. Download a pretrained DenseCap model by running the following script:
```bash
    cd densecap
    wget http://cs.stanford.edu/people/jcjohns/densecap/densecap-pretrained-vgg16.t7.zip
    unzip densecap-pretrained-vgg16.t7.zip
    rm densecap-pretrained-vgg16.t7.zip
```
### Preprocess Data
To extract 50 image region features from each image, use the following command: 
```bash
    cd densecap
    th extract_features.lua
```
Then do:
```bash
    python scripts/prepro_labels.py
```
`prepro_labels.py` will create `data/paratalk.json` and `data/paratalk.h5`. The `paratalk.json` contains the image information, while the `paratalk.h5` includes the preprocessed image paragraph captions and dense captions.

At last, preprocess ngrams for computing the reward:
```bash
    python scripts/prepro_ngrams.py
```

## Training
```bash
    python train.py --input_json data/paratalk.json  --input_label_h5 data/paratalk_label.h5   --input_feature data/VG_feature.h5
```
The training process contains two steps:
1. We first pre-train the model under the cross-entropy cost.
2. Then, the self-critical training method with CIDEr as the reward is used to further optimize the model.

## Evaluation
Run the eval script:
```bash
    python eval.py --model_path save_scst/model-best.pth --infos_path save_scst/infos-best.pkl 
```

