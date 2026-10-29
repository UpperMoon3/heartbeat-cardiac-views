# Which cardiac views carry the signal for congenital heart disease? — analysis code

Code for the analyses of the manuscript on view contribution and acquisition bias in the second-trimester
partition of the public Heartbeat fetal echocardiography dataset.

## Data
The Heartbeat dataset is not redistributed. Request access from its authors (Rodríguez et al., 2026,
Frontiers in Cardiovascular Medicine, doi:10.3389/fcvm.2026.1726484) under their academic-use terms.
ImageNet ResNet-18 weights: torchvision `resnet18` ImageNet weights saved as a state dict.

## Run
```
pip install -r requirements.txt
cd src
python 01_features.py /path/to/Heartbeat/dataset resnet18_imagenet.pth features.pkl
python 02_scores.py features.pkl scores.pkl
python 03_analysis.py scores.pkl results_main.json
python 04_official_and_sensitivity.py features.pkl /path/to/Heartbeat/dataset results_official.json
```
All splits are grouped by patient. Classifier hyperparameters are fixed a priori in `common.py`.
