# Real-Time-Emotion-Recognition-Using-Deep-Learning

## Overview

This project utilizes deep learning techniques to recognize emotions from facial expressions in real-time. The application leverages a Convolutional Neural Network (CNN) model trained on facial emotion datasets to classify emotions such as angry, disgust, fear, happy, neutral, sad, and surprise.

## Features

- Real-time emotion detection using webcam feed.
- Emotion prediction from uploaded images.
- Confidence score for each predicted emotion.
- User-friendly interface built with Streamlit.

## Technologies Used

- Python
- Streamlit
- OpenCV
- Keras
- NumPy
- Pillow

## Installation

1. Install the required packages:

pip install -r requirements.txt

2. Download the model files (facialemotionmodel.json and facialemotionmodel_weights.h5) and place them in the project directory.

3. Obtain the dataset from Kaggle and follow the instructions to include it in the project if necessary.

## Usage
1. Run the Streamlit app:
streamlit run app.py

2. Open your web browser and go to http://localhost:8501 to access the application.

3. Use the "Open Camera" button to start the webcam feed or upload an image to predict the emotion.

## How It Works
- The application captures frames from the webcam or processes uploaded images.
- Each frame/image is preprocessed (converted to grayscale, resized, normalized) before being fed into the trained CNN model.
- The model outputs the predicted emotion along with a confidence score, which is displayed on the interface.

## Contributing
Contributions are welcome! If you have suggestions for improvements or new features, feel free to open an issue or submit a pull request.

## Acknowledgements

- Keras for the deep learning framework.
- OpenCV for computer vision functionalities.
- Streamlit for creating the web application interface.
- Dataset sourced from Kaggle (specify the dataset name or link if applicable).

## Contact
For any questions or inquiries, please reach out to yashwantmanish@gmail.com.

