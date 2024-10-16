import streamlit as st
import cv2
import numpy as np
import time
from keras.models import model_from_json
from PIL import Image

# Define emotion labels
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

# Load the model only once using Streamlit caching
@st.cache_resource
def load_emotion_model():
    with open('emotiondetector.json', 'r') as json_file:
        model_json = json_file.read()
    model = model_from_json(model_json)
    model.load_weights('emotiondetector.h5')
    return model

model = load_emotion_model()

# Function to preprocess the frame/image
def preprocess_frame(frame, target_size=(48, 48)):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    resized_frame = cv2.resize(gray_frame, target_size)  # Resize to the model's expected size
    normalized_frame = resized_frame / 255.0  # Normalize pixel values
    reshaped_frame = np.reshape(normalized_frame, (1, 48, 48, 1))  # Reshape for the model
    return reshaped_frame

# Function to make prediction and display emotion
def predict_emotion(frame):
    preprocessed_frame = preprocess_frame(frame)
    prediction = model.predict(preprocessed_frame)
    emotion_label_index = np.argmax(prediction)
    confidence = prediction[0][emotion_label_index]
    emotion = labels[emotion_label_index]
    return emotion, confidence

# Streamlit app UI
st.title('Facial Emotion Detection')

# Camera control buttons
if 'camera_on' not in st.session_state:
    st.session_state.camera_on = False

# Webcam video capture
cap = None

# Create two columns for buttons
col1, col2 = st.columns(2)

# Toggle camera buttons in the same line
with col1:
    if st.button("Open Camera"):
        st.session_state.camera_on = True
        cap = cv2.VideoCapture(0)  # Start video capture

with col2:
    if st.button("Close Camera"):
        st.session_state.camera_on = False
        if cap:
            cap.release()
            cv2.destroyAllWindows()

# Image upload section
uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

# Webcam and real-time emotion detection
if st.session_state.camera_on:
    stframe = st.empty()  # Placeholder for video frames
    if cap.isOpened():
        while True:
            ret, frame = cap.read()
            if not ret:
                st.write("Failed to grab frame.")
                break

            # Predict emotion from webcam feed
            emotion, confidence = predict_emotion(frame)

            # Display emotion and confidence on frame
            cv2.putText(frame, f"Emotion: {emotion} ({confidence:.2f})", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Display video feed in Streamlit
            stframe.image(frame, channels="BGR")

            # Add delay for smoother frame processing
            time.sleep(0.1)
else:
    st.write("Camera is off.")

# Emotion prediction from uploaded image
if uploaded_image is not None:
    # Convert the uploaded image to OpenCV format
    image = np.array(Image.open(uploaded_image))
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Convert the uploaded image to BGR for prediction (OpenCV uses BGR)
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # If grayscale image, convert to BGR
    else:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert from RGB to BGR

    # Predict emotion from the uploaded image
    emotion, confidence = predict_emotion(image)

    # Display the predicted emotion
    st.write(f"Predicted Emotion: {emotion} with confidence {confidence:.2f}")
