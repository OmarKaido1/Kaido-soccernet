# Action Spotting using Multi-Modal Learning Models

# Introduction

This project focuses on Action Spotting in football matches using multi-modal learning models that combine both visual and audio information. Traditional unimodal approaches (using only video or audio) often struggle to capture the complexity of sports events. Our model integrates features extracted from video frames (via ResNet and YOLO) and audio signals (via VGGish and FFT) to improve the accuracy and robustness of action spotting. The problem addressed is the automatic detection and classification of key football events (e.g., goals, fouls, substitutions), which is vital for analytics, broadcasting, and coaching applications.

# Requirements

Python 3.9+

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

# Installation

1. Clone the repository:

git clone https://github.com/OmarKaido1/Kaido-soccernet.git

cd Kaido-soccernet

2. Create a virtual environment and activate it:
python -m venv venv

source venv/bin/activate   # Linux/Mac  

venv\Scripts\activate      # Windows

3. Install dependencies:
pip install -r requirements.txt
# Publication
From Tracking to Action Recognition: A Deep Learning Framework for Football Action Spotting
Optimizing Football Action Spotting using Multi-Modal Learning Models
# Run

To run inference on a football video:


# Contact

Researcher: Omar Ahmed Kazem

Email: [oa0175888@gmail.com]
