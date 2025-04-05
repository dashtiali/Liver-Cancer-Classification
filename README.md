
![Liver Cancer Classification](https://img.shields.io/badge/Repository-Liver%20Cancer%20Classification-blue)

# 🧬 Liver Cancer Classification

![Visitors](https://visitor-badge.glitch.me/badge?page_id=dashtiali/Liver-Cancer-Classification)

This repository contains the scripts used in the experiments for the project on:

> 📜 [Leveraging Persistent Homology for Liver Tumour Classification](https://doi.org/10.1117/12.3045640)


## 📚 Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Cite this Paper](#citation)


## 🧠 Introduction

Distinguishing between intrahepatic cholangiocarcinoma (ICC) and hepatocellular carcinoma (HCC) in imaging is a difficult task for a radiologist. We endeavoured to develop reliable models to automatically classify these tumour types. 

In this study, we propose using **persistent homology (PH)**, from the field of **topological data analysis (TDA)**, to build topological shapes from computed tomography (CT) scans of the liver. PH extracts summaries such as persistent connected components and loops from CT scans in the form of persistent barcodes.

Using these topological features, we trained a variety of classifiers, achieving:

🎯 **97.5% accuracy** and **97.56% F1-score**

We also evaluated radiomics features and pre-trained CNN models. Results were comparable with TDA being marginally higher.


## 🚀 Features

- 📂 Data preprocessing for CT scans
- 🔍 Training 15+ classifiers with TDA & Radiomics features
- 🧠 Training multiple CNN architectures for benchmarking
- ⚙️ Hyperparameter tuning and evaluation
- 📊 Visualizations of performance metrics
- 📄 Easy-to-follow documentation


## 🛠 Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/dashtiali/liver-cancer-classification.git
cd liver-cancer-classification
pip install -r requirements.txt
```


## ▶️ Usage

To reproduce the full pipeline, run the following scripts **in order**:

1. 🧮 **Compute persistent barcodes** from slices of the preprocessed CT image:
   ```bash
   python compute_persistent_barcodes.py
   ```

2. 📐 **Vectorize** the persistent barcodes:
   ```bash
   python extract_features.py
   ```

3. 🧠 **Train** a variety of classification models:
   ```bash
   python classifications.py
   ```

4. 🎯 **Optimize hyperparameters** for improved performance:
   ```bash
   python hyperparameter_tuning.py
   ```


## 📈 Results

### Best Performing Classifier for Each Feature Type on the Testing Set

<div align="center">

| Feature                     | Classifier        | Accuracy | AUC   | Recall  | Precision | F1    |
|----------------------------|-------------------|:--------:|:-----:|:-------:|:---------:|:-----:|
| Betti Curve                | Extra Trees       | 92.50    | 97.00 | 95.00   | 90.48     | 92.68 |
| Entropy Summary            | Cat Boost         | 87.50    | 92.00 | 90.00   | 85.71     | 87.80 |
| **Pers. Landscape**        | **Cat Boost**     | **97.50**| 99.00 | **100.00**| **95.24** | **97.56** |
| Pers. Statistics           | LR                | 90.00    | 96.00 | 95.00   | 86.36     | 90.48 |
| Pers. Tropical Coordinates | CatBoost          | 87.50    | 91.50 | 90.00   | 85.71     | 87.80 |
| Radiomics First Order      | Extra Trees       | 87.50    | 97.00 | 85.00   | 89.47     | 87.18 |
| Radiomics GLCM             | LR                | 82.50    | 93.50 | 80.00   | 84.21     | 82.05 |
| Radiomics GLDM             | Extra Trees       | 95.00    | 98.50 | **100.00**| 90.91   | 95.24 |
| Radiomics GLRLM            | Gradient Boosting | 92.50    | 98.25 | 95.00   | 90.48     | 92.68 |
| Radiomics GLSZM            | Gradient Boosting | 95.00    | 98.50 | 95.00   | 95.00     | 95.00 |
| Radiomics NGTDM            | LR                | 87.50    | 95.75 | 85.00   | 89.47     | 87.18 |
| Radiomics Shape2D          | LR                | 75.00    | 82.00 | 70.00   | 77.78     | 73.68 |
| Deep Learning              | Resnet50          | 87.50    | 98.00 | **100.00**| 80.00   | 88.89 |
| Deep Learning              | Inception V3      | 95.00    | 99.25 | 95.00   | 95.00     | 95.00 |
| Deep Learning              | Xception          | 95.00    | **99.50**| 95.00 | 95.00     | 95.00 |

</div>


## 📖 Citation

If you found this repository useful and would like to cite the paper associated with this project, please use the following citation:

```bibtex
@inproceedings{10.1117/12.3045640,
  author = {Dashti A. Ali and Jacob J. Peoples and Ramtin Mojtahedi and Kaitlyn S. Kobayashi and William R. Jarnagin and Richard K. G. Do and Amber L. Simpson},
  title = {{Leveraging persistent homology for liver tumour classification}},
  volume = {13407},
  booktitle = {Medical Imaging 2025: Computer-Aided Diagnosis},
  editor = {Susan M. Astley and Axel Wism{"u}ller},
  organization = {International Society for Optics and Photonics},
  publisher = {SPIE},
  pages = {134071Q},
  keywords = {Topological data analysis, Persistent homology, Liver cancer},
  year = {2025},
  doi = {10.1117/12.3045640},
  URL = {https://doi.org/10.1117/12.3045640}
}
```

🧪 _Built with love for research, reproducibility, and innovation in medical imaging._
