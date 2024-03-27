from PIL import Image
import streamlit as st
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
import cv2
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns


# Initialize session state
if "initialized" not in st.session_state:
    st.session_state.initialized = True
    # Here you can initialize other session_state attributes if needed

# Load a subset of the MNIST dataset
mnist = fetch_openml('mnist_784', version=1, cache=True,
                     as_frame=False, parser='auto')

X = mnist["data"][:10000]  # Use a subset of the data for faster prototyping
y = mnist["target"].astype(np.uint8)[:10000]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Define the machine learning models
models = {
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC(),
    'KNN': KNeighborsClassifier()
}

# Train and evaluate each model
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'{name} Accuracy: {accuracy}')
    print(f'{name} Classification Report:\n{
          classification_report(y_test, y_pred)}')
    print(f'{name} Confusion Matrix:')
    print(confusion_matrix(y_test, y_pred))
    print()

# Plot the accuracies of the models
accuracies = [accuracy_score(y_test, model.predict(X_test))
              for model in models.values()]

plt.figure(figsize=(10, 6))
plt.bar(models.keys(), accuracies, color=['blue', 'orange', 'green'])
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Accuracy of Different Models')
plt.ylim(0.9, 1.0)
plt.show()

# Choose the best model based on the highest accuracy
best_model_name = max(models, key=lambda x: accuracy_score(
    y_test, models[x].predict(X_test)))
best_model = models[best_model_name]

# Plot the confusion matrix for the best model
y_pred_best = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred_best)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title(f'Confusion Matrix for Best Model ({best_model_name})')
plt.show()


# Define a class to process the webcam video stream
class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        return img


# Initialize Streamlit app
st.title('Handwritten Digit Recognition')

# Navigation options
nav_option = st.sidebar.radio(
    "Navigation", ["Take Photo", "Upload Image", "The Best Model"])

if nav_option == "Take Photo":
    st.write("Take Photo option selected")

    # Initialize the webcam
    webrtc_ctx = webrtc_streamer(
        key="example", video_processor_factory=VideoTransformer)

    # Capture and process the image
    if webrtc_ctx.video_transformer:
        img = webrtc_ctx.video_transformer.transform(
            webrtc_ctx.video_frame)

        # Process the image
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)

            # Display the captured image
            st.image(img, caption='Captured Image', use_column_width=True)

elif nav_option == "Upload Image":
    # Option to upload image
    uploaded_file = st.file_uploader(
        "Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

elif nav_option == "The Best Model":
    st.write(f"Best Model: {best_model_name}")
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    st.write(f"Accuracy on test data: {accuracy}")
    st.write(f"Classification Report: {classification_rep}")
