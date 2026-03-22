D3F-Det: Progressive Fine-Grained Feature Refinement and Reuse for Robust Small Object Detection in Aerial Imagery

This code repository is associated with the paper "D3F-Det: Progressive Fine-Grained Feature Refinement and Reuse for Robust Small Object Detection in Aerial Imagery", which is currently under review at The Visual Computer journal. Please cite our submitted paper if you use this code for your research.


## 1. Project Introduction
Small object detection in unmanned aerial vehicle (UAV) imagery is highly challenging, as fine-grained features are prone to severe degradation and underutilization throughout the detection pipeline. Traditional methods often suffer from irreversible spatial detail loss during downsampling, ineffective spatial-semantic alignment at the same scale, and progressive dilution of shallow features in multi-scale fusion.

To address these interconnected issues, we propose a unified real-time detection framework, D3F-Det, designed to preserve, enhance, and progressively reuse fine-grained information for robust small object detection in aerial scenarios. The framework consists of three innovative core modules: Dual-branch Detail-preserving Downsampling (DBDown), Dual-feature Multi-scale Collaboration (DFMS), and Three-stage Reuse Neck (3S-RN). Extensive experiments on multiple challenging aerial small object datasets validate the effectiveness of D3F-Det, achieving significant performance gains over baseline methods while maintaining lightweight parameters. The core model configuration file is DBDown+DFMS+3S-RN.yaml.


## 2. Software Dependencies
All required packages with fixed versions are listed in requirements.txt.
Install command: pip install -r requirements.txt


## 3. Dataset Preparation
Our experiments are conducted on three classic aerial small object detection benchmarks: VisDrone2019, AI-TOD, and TinyPerson.

Dataset Download Links:
- VisDrone2019: https://github.com/VisDrone/VisDrone-Dataset
- AI-TOD: https://github.com/jwwangchn/AI-TOD
- TinyPerson: https://github.com/ucas-vg/TinyPerson

Place the downloaded and unzipped datasets into the data/ folder with a categorized structure: visdrone/, ai-tod/, tinyperson/. For each dataset, open the corresponding yaml configuration file and modify the dataset path to your local absolute path to ensure normal training and validation.


## 4. Model Training
The project uses a unified training script train.py. For different datasets, only the corresponding dataset yaml file needs to be switched, and the model configuration file remains fixed.

Training Commands:
- Train on VisDrone2019: python train.py --config DBDown+DFMS+3S-RN.yaml --dataset visdrone.yaml
- Train on AI-TOD: python train.py --config DBDown+DFMS+3S-RN.yaml --dataset ai-tod.yaml
- Train on TinyPerson: python train.py --config DBDown+DFMS+3S-RN.yaml --dataset tinyperson.yaml

Trained model weights are automatically saved to the checkpoints/ directory, and training logs are stored in the logs/ directory.


## 5. Model Validation & Evaluation
Independent validation scripts are provided for each dataset to evaluate model performance and output visual results.

Validation Commands:
- Validate on VisDrone2019: python val_visdrone.py --weights checkpoints/best_model.pth --dataset visdrone.yaml
- Validate on AI-TOD: python val_ai-tod.py --weights checkpoints/best_model.pth --dataset ai-tod.yaml
- Validate on TinyPerson: python val_tinyperson.py --weights checkpoints/best_model.pth --dataset tinyperson.yaml

The validation process automatically outputs evaluation metrics such as mAP@0.5:0.95, and visual results and metric files are saved to the results/ directory.


## 6. DOI & Permanent Archive

## 6. DOI & Permanent Archive
Repository DOI:10.5281/zenodo.19162854

## 7. Contact
If you encounter any problems during code reproduction, please contact us via email: [Your Contact Email]
