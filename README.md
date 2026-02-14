# Interactive Brain Tumor Segmentation (iBraTS)

This repository contains the code and implementation details for accurately segmenting images of the human brain,
specifically targeting tumor diagnosis and treatment planning. The proposed method utilizes deep neural networks
with interactive segmentation techniques, aiming to improve accuracy
while minimizing user effort and computational requirements.

### Documents

### Table of Contents

1. [Abstract](#abstract)
2. [Key Features](#key-features)
3. [Installation](#installation)
4. [Simple Run](#simple-run)
5. [Experiments and Results](#experiments-and-results)
6. [License](#license)
7. [Acknowledgement](#license)

## Abstract

Accurately segmenting 3D images of the human brain is crucial for medical applications such as tumor diagnosis
and treatment planning. Deep neural networks have emerged as the most advanced automatic segmentation method,
but they may require modifications for clinical use. Interactive segmentation allows for greater accuracy
by incorporating user interactions, but current methods are not suitable for low-power systems and may require
significant user effort. This study utilizes a novel method that performs a coarse segmentation on a low-resolution
section of the target region, followed by a local refinement to restore lost resolution. To optimize efficiency,
morphological analysis is used to modify only areas that need updating while preserving previously obtained results
for other regions. Experimental results demonstrate that this method achieves more accurate results
with less user interaction and requires less computing power and time compared to other methods.
Additionally, this method shows acceptable generalization in tasks not encountered during the training phase.

## Key Features

- Deep learning-based approach for accurate brain image segmentation
- Interactive segmentation techniques to incorporate user interactions
- Coarse segmentation followed by local refinement to restore resolution
- Efficient computation using selective morphological analysis
- Reduced user effort and improved accuracy compared to existing methods
- Generalization capabilities beyond the training phase

## Installation

To set up the project, please follow these steps:

1. Clone this repository:

```bash
git clone https://github.com/ali-sedaghi/ML-Medical-Image-Analysis.git
```

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

1. Download the dataset (BraTS2020) from
[source link](https://drive.google.com/drive/folders/1-_V-q80bmAGUhvyWT13rca4eEL-C7oXC?usp=share_link)
and place it in the following directory:

```bash
data/datasets/BraTS/
```

1. Download the pretrained models (SegFormer, HRNet, ResNet) from
[source link](https://drive.google.com/drive/folders/1iS9GHo627a81gbF7tTFBLHZQmqIMXUht?usp=share_link)
and place it in the following directory:

```bash
data/weights/
```

1. To train the model, run the following command with:

```bash
python ...
```

1. To evaluate the model, run the following command with:

```bash
python ,..
```

Refer to the source code and [Docs](./docs/) for additional options and configurations.

## Simple Run on Kaggle and Colab

You can use our ready to use notebooks on Kaggle and Google Colab.

- Kaggle train and evaluation notebook: [[Link]](https://www.kaggle.com/code/alisedaghi/runner)
- Google Colab train and evaluation notebook:
[[Link]](https://drive.google.com/file/d/1x_5paAO4z3stNoaQx8s9vKgx-FHrouk0/view?usp=sharing)
- Kaggle evaluation notebook: [[Link]](https://www.kaggle.com/code/zohrehbodaghi/evaluate)

## Experiments and Results

The experimental results demonstrate that our proposed method achieves more accurate segmentation results
with less user interaction and requires reduced computing power and time compared to other methods.
The model also exhibits acceptable generalization capabilities beyond the training phase.

For detailed results, analysis, and trained models please refer to our
[Experiments Drive](https://drive.google.com/drive/folders/17bD0_BudCUKC-7Z7y7hEhKjZ0rQ0ZGBE?usp=share_link) and
[Dcos](./docs/).

## License

This project is licensed under the Apache License. See the [LICENSE](./LICENSE) file for details.

## Acknowledgement

The core article of this codebase follows:

```bibtex
@misc{ClickSEG2022,
  title = {ClickSEG: A Codebase for Click-Based Interactive Segmentation},
  author = {Chen, Xi and Zhao, Zhiyan and Zhang, Yilei and Duan, Manni and Qi, Donglian and Zhao, Hengshuang},
  year = {2022}
}
