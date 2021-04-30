# Realtime-Face-Mask-Detection-using-Deep-Learning

## Overview
This paper proposes a method to perform automatic and real-time face mask detection using various techniques such as deep learning and image processing. A deep learning model, based on MobileNetV2 architecture, has been trained on a dataset containing a large number of relevant images. This trained model can, then, be used to detect the presence or absence of face masks in images, videos as well as in real-time video stream. The trained model achieves very high accuracy with 99.52% training accuracy and 99% validation accuracy. Further, due to the use of MobileNetV2 architecture, the trained model is a very light weight model and can be easily and efficiently integrated even with mobile devices. The model has been created using the Keras framework of TensorFlow library and all image processing works have been done using the OpenCV library. Thus, due to automatic and accurate monitoring using the proposed method, transmission of various contagious diseases can be slowed down and prevented and safety of workers in industries and harmful environments can be enhanced.

## Structure of the project
- Dataset					 --> Contains all the images used to train the model.  
- Pretrained_face_detection_model_opencv	 --> Contains files necessary to perform face detection using OpenCV DNN.  
- mask_detector_model.h5			 --> Trained model to perform face mask detection.  
- train_model.ipynb				 --> Notebook to train the model.
- face_mask_detection_in_images.ipynb		 --> Notebook to perform face mask detection on images.  
- face_mask_detection_in_video.ipynb		 --> Notebook to perform face mask detection on local video.  
- face_mask_detection_in_webcam.ipynb		 --> Notebook to perform face mask detection on realtime video stream through webcam.

## Packages necessary
- Numpy
- Matplotlib
- TensorFlow
- OpenCV
- Scikit Learn
- Jupyter Notebook

## Execution instructions
1. Training the model (Can be skipped. Trained model is already included in the project code)  
   - Open the notebook: train_model.ipynb
   - Execute the entire notebook:  
     Cells > Run All  
     
2. Performing face mask detection on images
   - Open the notebook: face_mask_detection_in_images.ipynb
   - Execute the entire notebook:  
     Cells > Run All 
   - Browse the image on which face mask detection is to be performed using the GUI.
   - Click Submit and wait for the process to complete.
   - Output will be displayed.
   
3. Performing face mask detection on local videos
   - Open the notebook: face_mask_detection_in_video.ipynb
   - Change the path of the video in the notebook itself. (Can be skipped)
   - Execute the entire notebook:  
     Cells > Run All  
   - Output will be displayed.
   
4. Performing face mask detection on realtime video stream through webcam
   - Open the notebook: face_mask_detection_in_webcam.ipynb
   - Execute the entire notebook:  
     Cells > Run All 
   - This will start your webcam and perform real time face mask detection.  
   
## Purpose
This project has been created for partial fulfilment of the course IT290 (Seminar).

## Team Members
- Aprameya Dash
- Gaikwad Ekansh Arun

## Guide
Prof. Shrutilipi Bhattacharjee
