#thunder model
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
import imageio
from IPython.display import display, HTML, clear_output, Image
from google.colab.patches import cv2_imshow

# Load Thunder model from TF Hub
model = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")
movenet = model.signatures['serving_default']
input_size = 256  # Thunder model expects 256x256 input size

# Download sample GIF
!wget -q -O dance.gif https://github.com/tensorflow/tfjs-models/raw/master/pose-detection/assets/dance_input.gif

# Load the GIF file
image_path = 'dance.gif'
image = tf.io.read_file(image_path)
image = tf.image.decode_gif(image)

def draw_keypoints_edges(frame, keypoints, confidence_threshold=0.3):
    """
    Draw keypoints and connections on the frame
    Args:
        frame: Input image frame
        keypoints: Detected pose keypoints
        confidence_threshold: Minimum confidence score to display a keypoint
    Returns:
        Frame with drawn keypoints and connections
    """
    height, width, _ = frame.shape
    
    # Edge definitions for skeleton
    edges = {
        (0, 1): 'm',    # nose to left eye
        (0, 2): 'm',    # nose to right eye
        (1, 3): 'm',    # left eye to left ear
        (2, 4): 'm',    # right eye to right ear
        (0, 5): 'g',    # nose to left shoulder
        (0, 6): 'g',    # nose to right shoulder
        (5, 7): 'g',    # left shoulder to left elbow
        (7, 9): 'g',    # left elbow to left wrist
        (6, 8): 'g',    # right shoulder to right elbow
        (8, 10): 'g',   # right elbow to right wrist
        (5, 6): 'g',    # left shoulder to right shoulder
        (5, 11): 'b',   # left shoulder to left hip
        (6, 12): 'b',   # right shoulder to right hip
        (11, 12): 'b',  # left hip to right hip
        (11, 13): 'b',  # left hip to left knee
        (13, 15): 'b',  # left knee to left ankle
        (12, 14): 'b',  # right hip to right knee
        (14, 16): 'b'   # right knee to right ankle
    }
    
    # Color definitions for visualization
    colors = {
        'm': (255, 0, 255),   # Magenta for face connections
        'g': (0, 255, 0),     # Green for arms
        'b': (255, 165, 0)    # Orange for legs
    }
    
    # Convert normalized coordinates to pixel coordinates
    y_coords = keypoints[:, 0] * height
    x_coords = keypoints[:, 1] * width
    confidences = keypoints[:, 2]
    
    # Draw edges (skeleton connections)
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
            cv2.circle(frame, (int(x), int(y)), 6, (0, 255, 255), -1)
            # Display confidence score for high-confidence points
            if conf > 0.5:
                cv2.putText(frame, f'{conf:.2f}', (int(x), int(y)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return frame

def process_gif(display_progress=True):
    """
    Process the GIF file frame by frame
    Args:
        display_progress: Whether to display processing progress
    Returns:
        List of processed frames
    """
    num_frames = image.shape[0]
    processed_frames = []
    
    for i in range(num_frames):
        # Get current frame
        frame = image[i].numpy().astype(np.uint8)
        
        # Prepare image for model
        input_image = tf.expand_dims(frame, axis=0)
        input_image = tf.image.resize_with_pad(input_image, input_size, input_size)
        input_image = tf.cast(input_image, dtype=tf.int32)
        
        # Run pose detection
        results = movenet(input_image)
        keypoints = results['output_0'].numpy()[0, 0]
        
        # Draw keypoints on frame
        output_frame = draw_keypoints_edges(frame.copy(), keypoints)
        
        # Add frame number and model info
        cv2.putText(output_frame, f'Frame: {i}/{num_frames}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(output_frame, 'MoveNet Thunder', (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        processed_frames.append(output_frame)
        
        # Show progress
        if display_progress and i % 5 == 0:
            print(f"Processing frame: {i}/{num_frames} ({(i/num_frames*100):.1f}%)")
            cv2_imshow(output_frame)
            clear_output(wait=True)
    
    return processed_frames

# Process the GIF
print("Processing GIF...")
output_frames = process_gif()

# Save the processed GIF
print("Saving processed GIF...")
imageio.mimsave('processed_dance_thunder.gif', output_frames, fps=30)

# Display the processed GIF
display(Image(filename='processed_dance_thunder.gif'))
