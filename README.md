# Action Spotting using Multi-Modal Learning Models

# Introduction

This project focuses on Action Spotting in football matches using multi-modal learning models that combine both visual and audio information. Traditional unimodal approaches (using only video or audio) often struggle to capture the complexity of sports events. Our model integrates features extracted from video frames (via ResNet and YOLO) and audio signals (via VGGish and FFT) to improve the accuracy and robustness of action spotting. The problem addressed is the automatic detection and classification of key football events (e.g., goals, fouls, substitutions), which is vital for analytics, broadcasting, and coaching applications.

# Requirements

Python 3.8+

PyTorch / TensorFlow (depending on implementation)

OpenCV for video preprocessing

Librosa for audio feature extraction

models: ResNet-152, YOLOv5, VGGish
 
requirements.txt file for installing all dependencies

# How to download SoccerNet
A SoccerNet pip package to easily download the data and the annotations is available.

To install the pip package simply run:

pip install SoccerNet

Please follow the instructions provided in the Download folder of this repository. Do also mind that signing an Non-Disclosure agreement (NDA) is required to access the LQ and HQ videos: NDA.

### Setup and Installation

First, you need to create a conda environment with the necessary dependencies. You can do this by running the following commands:

```bash
conda create --name SoccerNet python=3.8.10
conda activate SoccerNet
conda install cudnn cudatoolkit=10.1
pip install scikit-video tensorflow==2.3.0 imutils opencv-python==3.4.11.41 SoccerNet moviepy scikit-learn ffmpy resampy
```

-----

### Running the Scripts

Here are the commands to run the various scripts in the repository.

#### 1\. Extract ResNET Features

This script extracts features for all 550 games in the SoccerNet-v2 dataset.

```bash
python Features/ExtractResNET_TF2.py \
--soccernet_dirpath "PATH/TO/SOCCERNET/DATASET" \
--back_end=TF2 \
--features=ResNET \
--video LQ \
--transform crop \
--verbose \
--split all
```

**Arguments:**

  * `--soccernet_dirpath`: Path to the SoccerNet directory.
  * `--back_end`: Backend for the model (e.g., TF2).
  * `--features`: The features to extract (e.g., ResNET).
  * `--video`: The video quality to use, either "LQ" or "HQ".
  * `--transform`: The transformation to apply, either "crop" or "resize".
  * `--verbose`: Enables verbose output.
  * `--split`: The split of videos from SoccerNet (e.g., all).

#### 2\. Reduce Features with PCA

This script reduces the features for all 550 games using PCA.

```bash
python Features/ReduceFeaturesPCA.py --soccernet_dirpath "PATH/TO/SOCCERNET/DATASET"
```

**Arguments:**

  * `--soccernet_dirpath`: Path to the SoccerNet directory.
  * `--features`: The features to perform PCA on, default is `ResNET_TF2.npy`.
  * `--features_PCA`: The name of the reduced features file, default is `ResNET_TF2_PCA512.npy`.
  * `--pca_file`: The pickle file for PCA, default is `pca_512_TF2.pkl`.
  * `--scaler_file`: The pickle for the average, default is `average_512_TF2.pkl`.

-----

### 4\. Extract Audio Features

This script extracts audio features from videos and processes them using a VGGish model.

```bash
python audio-features-extraction/extract_features.py
```

The script can be configured by modifying the `ROOT_PATH` and `OUT_DIR_PATH` variables within the file itself. It will iterate through subdirectories in `ROOT_PATH`, convert `.mkv` files to `.wav`, and extract features, saving them to `OUT_DIR_PATH`.

-----

### 3\. Evaluate Action Spotting

This script is used to evaluate the performance of action spotting predictions.

```bash
python Evaluation/EvaluateSpotting.py \
--SoccerNet_path "PATH/TO/SOCCERNET/DATASET" \
--Predictions_path "PATH/TO/PREDICTIONS" \
--Prediction_file "Predictions-v2.json"
```

**Arguments:**

  * `--SoccerNet_path`: Path to the SoccerNet-V2 dataset folder (or zipped file) with labels.
  * `--Predictions_path`: Path to the predictions folder (or zipped file) with predictions.
  * `--Prediction_file`: The name of the prediction files as stored in the folder.
  * `--split`: The set on which to evaluate the performances, default is "test".
  * `--version`: The version of SoccerNet, default is 2.

-----
# PUBLICATIONS
From Tracking to Action Recognition: A Deep Learning Framework for Football Action Spotting

Optimizing Football Action Spotting using Multi-Modal Learning Models
# Contact

Researcher: Omar Ahmed Kazem

Email: [oa0175888@gmail.com]
