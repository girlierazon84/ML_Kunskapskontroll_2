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

if "initialized" not in st.session_state:
    st.session_state.initialized = True
    # Här kan du initialisera andra session_state-attribut om det behövs

# Ladda in MNIST-datasetet
mnist = fetch_openml('mnist_784', version=1, cache=True,
                     as_frame=False, parser='auto')

X = mnist["data"]
y = mnist["target"].astype(np.uint8)

# Dela upp datan i tränings- och testuppsättningar
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Träningsmodeller
models = {
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC(),
    'KNN': KNeighborsClassifier()
}

for name, model in models.items():
    model.fit(X_train, y_train)

    # Förutsäg på testuppsättningen
    y_pred = model.predict(X_test)

    # Utvärdera modellen
    accuracy = accuracy_score(y_test, y_pred)
    print(f'{name} Accuracy: {accuracy}')
    print(f'{name} Classification Report:\n{
          classification_report(y_test, y_pred)}')
    print(f'{name} Confusion Matrix:')
    print(confusion_matrix(y_test, y_pred))
    print()

    # Träningsmodeller
models = {
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC(),
    'KNN': KNeighborsClassifier()
}

accuracies = []

for name, model in models.items():
    model.fit(X_train, y_train)

    # Förutsäg på testuppsättningen
    y_pred = model.predict(X_test)

    # Utvärdera modellen
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

# Plotta noggrannheten för varje modell
plt.figure(figsize=(10, 6))
plt.bar(models.keys(), accuracies, color=['blue', 'orange', 'green'])
plt.xlabel('Modell')
plt.ylabel('Noggrannhet')
plt.title('Noggrannhet för olika träningsmodeller')
plt.ylim(0.9, 1.0)
plt.show()


# Välj den bästa modellen baserat på valfri utvärderingsmetod, t.ex. högsta noggrannhet
best_model_name = None
best_accuracy = 0

for name, model in models.items():
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    if accuracy > best_accuracy:
        best_model_name = name
        best_model = model
        best_accuracy = accuracy

print(best_model_name)
# Visa eller plotta förvirringsmatrisen för den bästa modellen
y_pred_best = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred_best)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title(f'Confusion Matrix for Best Model ({best_model_name})')
plt.show()


# =============================================================================
# Streamlit HDR_app.py code
# =============================================================================


# Train SVM model
model = SVC()
model.fit(X_train, y_train)


# Define a class to process the webcam video stream
class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        return img


def main():
    st.title('Handwritten Digit Recognition')

    # Navigation option
    nav_option = st.sidebar.radio(
        "Navigation", ["Take Photo", "Upload Image"])

    if nav_option == "Take Photo":
        st.write("Take Photo option selected")

        # Initialize the webcam
        webrtc_ctx = webrtc_streamer(
            key="example", video_transformer_factory=VideoTransformer)

        # Capture and process the image
        if webrtc_ctx.video_transformer:
            img = webrtc_ctx.video_transformer.get_frame()

            # Process the image
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)

                # Display the captured image
                st.image(img, caption='Captured Image', use_column_width=True)

        def preprocess_image(image):
            # Convert the image to grayscale
            gray = image.convert('L')
            # Resize the image to 28x28 pixels
            resized = gray.resize((28, 28))
            # Convert image to array and normalize
            image_array = np.array(resized) / 255.0
            # Flatten the array
            flattened_array = image_array.flatten()
            # Reshape to match model input shape
            reshaped_array = flattened_array.reshape(1, -1)
            return reshaped_array

        def predict_image(image):
            processed_image = preprocess_image(image)
            prediction = model.predict(processed_image)
            return prediction[0]

    elif nav_option == "Upload Image":
        # Option to upload image
        uploaded_file = st.file_uploader(
            "Choose an image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)

            # Preprocess the image
            processed_image = np.array(image)

            def preprocess_image(image):
                # Convert the image to grayscale
                gray = image.convert('L')
                # Resize the image to 28x28 pixels
                resized = gray.resize((28, 28))
                # Convert image to array and normalize
                image_array = np.array(resized) / 255.0
                # Flatten the array
                flattened_array = image_array.flatten()
                # Reshape to match model input shape
                reshaped_array = flattened_array.reshape(1, -1)
                return reshaped_array

            def predict_image(image):
                processed_image = preprocess_image(image)
                prediction = model.predict(processed_image)
                return prediction[0]

    elif nav_option == "The Best Model":
        st.write(f"Best Model: {best_model_name}")
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        classification_rep = classification_report(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)

        st.write(f"Accuracy on test data: {accuracy}")
        st.write(f"Classification Report: {classification_rep}")
        st.write(f"Confusion Matrix: {conf_matrix}")


if __name__ == "__main__":
    main()
