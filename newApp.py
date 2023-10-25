import os
import torch
import cv2 as cv
import numpy as np
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
from PIL import Image
from ultralytics import YOLO

app = Flask(__name__)

# Ensure the existence of the folders to store detected images and video frames
os.makedirs('detected_images', exist_ok=True)
os.makedirs('detected_video_frames', exist_ok=True)

@app.route("/", methods=['POST'])
def predict_img():
    if request.method == 'POST':
        if 'file' in request.files:
            f = request.files['file']
            file_extension = f.filename.rsplit('.', 1)[1].lower()

            if file_extension in ['jpg', 'jpeg', 'png']:
                # For images
                img = cv.imread(f)
                yolo = YOLO('yolov8n.pt')
                detections = yolo.predict(img, save=True)

                # Save detected image in the folder
                img_path = os.path.join('detected_images', secure_filename(f.filename))
                Image.fromarray(cv.cvtColor(detections[0].plot(), cv.COLOR_BGR2RGB)).save(img_path)

                # Prepare response data as a dictionary
                response_data = {
                    'message': 'Image detected and saved successfully',
                    'bounding_boxes': detections[0].pred[:, :4].tolist(),
                    'labels': detections[0].names,
                    'confidence_scores': detections[0].pred[:, 4].tolist()
                }

                return jsonify(response_data)

            elif file_extension == 'mp4':
                # For videos
                video_path = os.path.join('detected_video_frames', secure_filename(f.filename))
                f.save(video_path)

                # Initialize the yolov8 model here
                model = YOLO('yolov8n.pt')
                cap = cv.VideoCapture(video_path)

                frame_number = 0
                detected_frames = []

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Do yolo detection on the frame here
                    results = model(frame, save=True)

                    # Prepare data for detected frame
                    frame_data = {
                        'frame_number': frame_number,
                        'bounding_boxes': results[0].pred[:, :4].tolist(),
                        'labels': results[0].names,
                        'confidence_scores': results[0].pred[:, 4].tolist()
                    }
                    detected_frames.append(frame_data)

                    # Save detected video frame in the folder
                    frame_path = os.path.join('detected_video_frames', f'frame_{frame_number}.jpg')
                    Image.fromarray(cv.cvtColor(results[0].plot(), cv.COLOR_BGR2RGB)).save(frame_path)
                    frame_number += 1

                cap.release()

                # Prepare response data as a dictionary
                response_data = {
                    'message': 'Video detected and frames saved successfully',
                    'detected_frames': detected_frames
                }

                return jsonify(response_data)

if __name__ == '__main__':
    app.run(debug=True)
