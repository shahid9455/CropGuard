import streamlit as st
import tensorflow as tf
import numpy as np
from keras.preprocessing import image
from openai import OpenAI

# Load your pre-trained model
model = tf.keras.models.load_model('crop_disease_model.h5')  # Update with your model path

# Set up Solar LLM with Upstage
client = OpenAI(
    api_key="up_g41Ecn1SmjMBCwzVrsNX5lP6kVb5i",  # Your actual API key
    base_url="https://api.upstage.ai/v1/solar"
)

# Define function to predict disease from image
def predict_disease(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize the image
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)
    return predicted_class[0]  # Return the predicted class index

# Define function to get instructions from Solar LLM
def get_disease_advice(disease_name):
    stream = client.chat.completions.create(
        model="solar-pro",
        messages=[
            {
                "role": "system",
                "content": "You are an assistant helping farmers with crop diseases."
            },
            {
                "role": "user",
                "content": f"I have detected {disease_name} in my crop. How can I prevent it from spreading?"
            }
        ],
        stream=False,
    )
    return stream.choices[0].message.content

# Streamlit UI for the crop disease prediction system
st.title("Crop Disease Prediction and Prevention Assistant")

# Image uploader
uploaded_file = st.file_uploader("Upload an image of your crop", type=["jpg", "png", "jpeg"])

# Label dictionary for the diseases
label_dict = {
    0: "Pepper Bacterial Spot",
    1: "Pepper Healthy",
    2: "Potato Early Blight",
    3: "Potato Late Blight",
    4: "Potato Healthy",
    5: "Tomato Bacterial Spot",
    6: "Tomato Early Blight",
    7: "Tomato Late Blight",
    8: "Tomato Leaf Mold",
    9: "Tomato Septoria Leaf Spot",
    10: "Tomato Spider Mites",
    11: "Tomato Target Spot",
    12: "Tomato YellowLeaf Curl Virus",
    13: "Tomato Mosaic Virus",
    14: "Tomato Healthy"
}

if uploaded_file is not None:
    # Save the uploaded file
    with open("temp_image.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Predict disease
    predicted_class = predict_disease("temp_image.jpg")
    predicted_disease = label_dict[predicted_class]

    st.write(f"Predicted Disease: {predicted_disease}")
    
    # Confirm prediction with the user
    confirm = st.button(f"Confirm {predicted_disease}")

    if confirm:
        # Get disease prevention advice from Solar LLM
        advice = get_disease_advice(predicted_disease)
        st.write(f"Prevention Tips for {predicted_disease}:")
        st.write(advice)
