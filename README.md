# MoveNet Pose Detection

Real-time human pose detection using TensorFlow's MoveNet model. This project provides an easy-to-use implementation for detecting and tracking human poses in images and videos.

![Demo GIF](./demo.gif)

## Features

- Real-time pose detection
- Support for both image and video processing
- Multiple pose keypoint visualization options
- Easy-to-use interface
- Support for both Lightning and Thunder models

## Requirements

- tensorflow >= 2.13.0
- tensorflow-hub >= 0.14.0
- opencv-python >= 4.8.0
- numpy >= 1.24.3
- imageio >= 2.31.1


## Installation

1. Clone the repository

- git clone https://github.com/yourusername/movenet-pose-detection.git
- cd movenet-pose-detection


2. Install dependencies
- pip install -r requirements.txt

## Usage

### Basic Usage

- import tensorflow as tf
- import tensorflow_hub as hub
- import cv2
- import numpy as np

# Load model
- model = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
- movenet = model.signatures['serving_default']

# Process image
- image = tf.io.read_file('your_image.jpg')
- image = tf.image.decode_jpeg(image)


### Processing a Video

# Import required modules
- from pose_detector import MoveNetDetector

# Initialize detector
- detector = MoveNetDetector(model_type="lightning")  # or "thunder"

# Process video
- detector.process_video('input_video.mp4', 'output_video.mp4')


## Models

### Lightning Model
- Faster inference (30+ FPS)
- Lower accuracy
- Better for real-time applications

### Thunder Model
- Slower inference (10-15 FPS)
- Higher accuracy
- Better for precise pose detection

## Keypoint Mapping

The model detects 17 keypoints:
- 0: nose
- 1: left eye
- 2: right eye
- 3: left ear
- 4: right ear
- 5: left shoulder
- 6: right shoulder
- 7: left elbow
- 8: right elbow
- 9: left wrist
- 10: right wrist
- 11: left hip
- 12: right hip
- 13: left knee
- 14: right knee
- 15: left ankle
- 16: right ankle

## Acknowledgments

- TensorFlow team for the MoveNet model
- Google's TensorFlow Hub for model distribution
- Original MoveNet paper authors
