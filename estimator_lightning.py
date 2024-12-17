#lightning model
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
import imageio
from IPython.display import display, HTML
from google.colab.patches import cv2_imshow
from google.colab import output

# Load MoveNet model
model_name = "movenet_lightning"
model = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
movenet = model.signatures['serving_default']

# Download GIF
!wget -q -O dance.gif https://github.com/tensorflow/tfjs-models/raw/master/pose-detection/assets/dance_input.gif

# Load GIF
image_path = 'dance.gif'
image = tf.io.read_file(image_path)
image = tf.image.decode_gif(image)

def draw_keypoints_edges(frame, keypoints, confidence_threshold=0.3):
    """Draw keypoints and connections"""
    height, width, _ = frame.shape
    
    # Edge definitions
    edges = {
        (0, 1): 'r',    # nose - left eye
        (0, 2): 'r',    # nose - right eye
        (1, 3): 'r',    # left eye - left ear
        (2, 4): 'r',    # right eye - right ear
        (0, 5): 'g',    # nose - left shoulder
        (0, 6): 'g',    # nose - right shoulder
        (5, 7): 'g',    # left shoulder - left elbow
        (7, 9): 'g',    # left elbow - left wrist
        (6, 8): 'g',    # right shoulder - right elbow
        (8, 10): 'g',   # right elbow - right wrist
        (5, 6): 'g',    # left shoulder - right shoulder
        (5, 11): 'b',   # left shoulder - left hip
        (6, 12): 'b',   # right shoulder - right hip
        (11, 12): 'b',  # left hip - right hip
        (11, 13): 'b',  # left hip - left knee
        (13, 15): 'b',  # left knee - left ankle
        (12, 14): 'b',  # right hip - right knee
        (14, 16): 'b'   # right knee - right ankle
    }
    
    # Color definitions
    colors = {
        'r': (255, 0, 0),    # Red
        'g': (0, 255, 0),    # Green
        'b': (0, 0, 255)     # Blue
    }
    
    # Convert coordinates
    y_coords = keypoints[:, 0] * height
    x_coords = keypoints[:, 1] * width
    confidences = keypoints[:, 2]
    
    # Draw edges
    for edge, color_code in edges.items():
        p1, p2 = edge
        y1, x1, c1 = y_coords[p1], x_coords[p1], confidences[p1]
        y2, x2, c2 = y_coords[p2], x_coords[p2], confidences[p2]
        
        if c1 > confidence_threshold and c2 > confidence_threshold:
            cv2.line(frame, 
                    (int(x1), int(y1)), 
                    (int(x2), int(y2)), 
                    colors[color_code], 
                    2)
    
    # Draw keypoints
    for i, (x, y, conf) in enumerate(zip(x_coords, y_coords, confidences)):
        if conf > confidence_threshold:
            cv2.circle(frame, (int(x), int(y)), 4, (255, 255, 0), -1)
    
    return frame

def process_gif():
    num_frames = image.shape[0]
    processed_frames = []
    input_size = 192  # For Lightning model

    for i in range(num_frames):
        # Get frame
        frame = image[i].numpy().astype(np.uint8)
        
        # Prepare image for model
        input_image = tf.expand_dims(frame, axis=0)
        input_image = tf.image.resize_with_pad(input_image, input_size, input_size)
        input_image = tf.cast(input_image, dtype=tf.int32)
        
        # Pose detection
        results = movenet(input_image)
        keypoints = results['output_0'].numpy()[0, 0]
        
        # Draw keypoints
        output_frame = draw_keypoints_edges(frame.copy(), keypoints)
        processed_frames.append(output_frame)
        
        # Show progress
        if i % 10 == 0:
            print(f"Processing frame: {i}/{num_frames}")
    
    return processed_frames

# Process GIF
print("Processing GIF...")
output_frames = process_gif()

# Save processed GIF
print("Saving processed GIF...")
imageio.mimsave('processed_dance_lightning.gif', output_frames, fps=30)

# Display processed GIF
from IPython.display import Image
display(Image(filename='processed_dance_lightning.gif'))
