import streamlit as st
from openai import OpenAI
import random
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from keras.preprocessing import image

# Initialize the OpenAI client
client = OpenAI(
    api_key="Add your AIMLAPI key here",
    base_url="https://api.aimlapi.com",
)

client_solar = OpenAI(
    api_key="Add your solar api key",  # Solar API key
    base_url="https://api.upstage.ai/v1/solar"
)
# Load your pre-trained crop disease model
model_disease = tf.keras.models.load_model('crop_disease_model.h5')  # Update with your model path

# Load your pre-trained crop recommendation model
model_recommendation = tf.keras.models.load_model('crop_recommendation_model.h5')

# Load the dataset for crop recommendation
dataset = pd.read_csv('Crop_recommendation.csv')

# Features and labels for recommendation system
X = dataset[['temperature', 'humidity', 'ph', 'water availability', 'season']].copy()
y = dataset['label']

# Encode the 'season' and 'label' columns
label_encoder_season = LabelEncoder()
label_encoder_season.fit(X['season'])
label_encoder_crop = LabelEncoder()
label_encoder_crop.fit(y)

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(
    X[['temperature', 'humidity', 'ph', 'water availability', 'season']].replace(
        label_encoder_season.classes_, label_encoder_season.transform(label_encoder_season.classes_)
    )
)

# List of short responses for non-crop-related questions
non_crop_responses = [
    "I'm here to assist with crop-related inquiries. Please ask about crops or agriculture.",
    "For crop-related questions, I'm your go-to assistant. Ask me about crops!",
    "Let's stick to crop and agriculture questions. How can I assist you with those?",
    "I specialize in crops and agriculture. Ask me anything related to these topics!",
    "Please ask about crops or farming practices. I can help with that!"
]

# A function to check if the prompt is related to crops or agriculture
def is_crop_related(prompt):
    crop_keywords = ['crop', 'plant', 'disease', 'farm', 'agriculture', 'soil', 'fertilizer', 'pesticide', 'harvest', 'irrigation', 'seed']
    prompt = prompt.lower()
    return any(keyword in prompt for keyword in crop_keywords)

# Function to get the AI response
def get_chat_response(prompt, crop_related=True):
    if crop_related:
        system_message = (
            "You are CropGuard, an AI assistant specialized in crop management, "
            "crop diseases, healthy plant practices, and crop-related advice. "
            "You will only answer questions related to crops and agriculture."
        )
    else:
        return random.choice(non_crop_responses)

    response = client.chat.completions.create(
        model="meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1500
    )
    message = response.choices[0].message.content
    return message

# Define function to predict disease from image
def predict_disease(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize the image
    predictions = model_disease.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)
    return predicted_class[0]  # Return the predicted class index

# Define function to get information about a disease
def get_disease_info(disease_name):
    response = client.chat.completions.create(
        model="meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
        messages=[
            {"role": "system", "content": "You are an expert on crop diseases and agriculture."},
            {"role": "user", "content": f"Provide comprehensive details about {disease_name}. Include introduction, causes, prevention methods, danger level, recommended pesticides, and any images if available."}
        ],
        max_tokens=2000
    )
    return response.choices[0].message.content

# Define function to get advice for healthy crops
def get_healthy_advice():
    response = client.chat.completions.create(
        model="meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
        messages=[
            {"role": "system", "content": "You are an expert on crop care and agriculture."},
            {"role": "user", "content": "My crop is healthy. How can I ensure it remains healthy and prevent diseases?"}
        ],
        max_tokens=1000
    )
    return response.choices[0].message.content

# Define function to handle form submission for the chat system
def handle_submit(user_input):
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
    
        # Check if the input is crop-related
        crop_related = is_crop_related(user_input)
    
        # Get AI response
        response = get_chat_response(user_input, crop_related)
    
        # Add AI response to session state
        st.session_state.messages.append({"role": "assistant", "content": response})

# Define function to predict crop recommendation
def predict_crop(temp, hum, ph, water, season):
    # Encode the season input
    season_encoded = label_encoder_season.transform([season])[0]
    
    # Prepare the input array
    user_input = np.array([[temp, hum, ph, water, season_encoded]])
    
    # Scale the input
    user_input_scaled = scaler.transform(user_input)
    
    # Make prediction
    prediction = model_recommendation.predict(user_input_scaled)
    
    # Get the predicted class with the highest probability
    predicted_class = np.argmax(prediction, axis=1)
    
    # Decode the predicted class back to crop label
    predicted_crop = label_encoder_crop.inverse_transform(predicted_class)
    
    return predicted_crop[0]

# Main Streamlit app function
def main():
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", ["Crop Disease Prediction", "Chat with CropGuard", "Crop Recommendation"])

    if selection == "Crop Disease Prediction":
        st.title("CropGuard: You Crop Disease Prediction AI Agent")

        # Image uploader
        uploaded_file = st.file_uploader("Upload an image of your crop", type=["jpg", "png", "jpeg"])

        # Label dictionary for the diseases
# Updated label dictionary with all disease classes
        label_dict = {
            0: 'bacterial_blight in Cotton',
            1: 'Corn___Northern_Leaf_Blight',
            2: 'RedRust sugarcane',
            3: 'Grape___healthy',
            4: 'Healthy Maize',
            5: 'Orange___Haunglongbing_(Citrus_greening)',
            6: 'Wheat___Yellow_Rust',
            7: 'Pepper__bell___Bacterial_spot',
            8: 'Tungro',
            9: 'Soybean___healthy',
            10: 'Wheat mite',
            11: 'Anthracnose on Cotton',
            12: 'Healthy Wheat',
            13: 'Squash___Powdery_mildew',
            14: 'Cotton Aphid',
            15: 'Common_Rust',
            16: 'Background_without_leaves',
            17: 'Potato___healthy',
            18: 'American Bollworm on Cotton',
            19: 'fresh cotton plant',
            20: 'Tomato_Leaf_Mold',
            21: 'Yellow Rust Sugarcane',
            22: 'Flag Smut',
            23: 'Tomato__Tomato_YellowLeaf__Curl_Virus',
            24: 'Corn___healthy',
            25: 'fresh cotton leaf',
            26: 'Wheat scab',
            27: 'Strawberry___Leaf_scorch',
            28: 'Army worm',
            29: 'cotton whitefly',
            30: 'Peach___healthy',
            31: 'Wheat leaf blight',
            32: 'Healthy cotton',
            33: 'Wilt',
            34: 'Tomato_Bacterial_spot',
            35: 'bollrot on Cotton',
            36: 'Apple___Apple_scab',
            37: 'Rice Blast',
            38: 'Becterial Blight in Rice',
            39: 'Tomato_Septoria_leaf_spot',
            40: 'Tomato_healthy',
            41: 'diseased cotton plant',
            42: 'cotton mealy bug',
            43: 'maize ear rot',
            44: 'Tomato_Spider_mites_Two_spotted_spider_mite',
            45: 'Tomato_Early_blight',
            46: 'Apple___Black_rot',
            47: 'Wheat Stem fly',
            48: 'Blueberry___healthy',
            49: 'Cherry___Powdery_mildew',
            50: 'Peach___Bacterial_spot',
            51: 'Tomato__Target_Spot',
            52: 'Apple___Cedar_apple_rust',
            53: 'Tomato___Target_Spot',
            54: 'Mosaic sugarcane',
            55: 'Sugarcane Healthy',
            56: 'Pepper__bell___healthy',
            57: 'red cotton bug',
            58: 'Pepper,_bell___healthy',
            59: 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
            60: 'Potato___Late_blight',
            61: 'Corn___Cercospora_leaf_spot Gray_leaf_spot',
            62: 'maize stem borer',
            63: 'Brownspot',
            64: 'bollworm on Cotton',
            65: 'pink bollworm in cotton',
            66: 'Strawberry___healthy',
            67: 'Leaf Curl',
            68: 'Corn___Common_rust',
            69: 'Apple___healthy',
            70: 'Grape___Black_rot',
            71: 'Wheat aphid',
            72: 'Tomato_Late_blight',
            73: 'diseased cotton leaf',
            74: 'Potato___Early_blight',
            75: 'maize fall armyworm',
            76: 'Wheat Brown leaf Rust',
            77: 'Leaf smut',
            78: 'Grape___Esca_(Black_Measles)',
            79: 'Wheat black rust',
            80: 'Raspberry___healthy',
            81: 'thirps on cotton',
            82: 'Tomato__Tomato_mosaic_virus',
            83: 'Cherry___healthy',
            84: 'RedRot sugarcane',
            85: 'Tomato___Spider_mites Two-spotted_spider_mite',
            86: 'Pepper,_bell___Bacterial_spot',
            87: 'Gray_Leaf_Spot',
            88: 'Wheat powdery mildew'
        }

# The rest of the code remains the same

        if uploaded_file is not None:
            # Save the uploaded file
            with open("temp_image.jpg", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Display the image in the center
            st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
            st.image("temp_image.jpg", caption='Uploaded Image', width=256)
            st.markdown("</div>", unsafe_allow_html=True)

            # Predict disease
            predicted_class = predict_disease("temp_image.jpg")
            predicted_disease = label_dict.get(predicted_class, "Unknown")

            if predicted_disease == "Unknown":
                st.write("The image does not seem to be of a plant. Please upload a valid image of a crop.")
            elif "Healthy" in predicted_disease:
                # Provide advice for healthy crops
                advice = get_healthy_advice()
                st.write(f"The crop is healthy: {predicted_disease}")
                st.write("Best practices to maintain health and prevent diseases:")
                st.write(advice)
            else:
                st.write(f"Predicted Disease: {predicted_disease}")
                
                # Fetch detailed information about the disease
                with st.expander(f"Details about {predicted_disease}"):
                    disease_info = get_disease_info(predicted_disease)
                    st.write(disease_info)

    elif selection == "Chat with CropGuard":
        st.title("CropGuard: Discuss about your crops and agriculture")

        # Initialize session state for messages if not present
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Use Streamlit's form to handle input and submission
        with st.form(key='chat_form', clear_on_submit=True):
            user_input = st.text_input(
                "Ask a question (preferably about crops):",
                key='user_input'
            )
            submit_button = st.form_submit_button(label='Submit')

        # Call the submit handler when the form is submitted
        if submit_button and user_input:
            handle_submit(user_input)

        # Display the chat history with user input at the top and previous messages below
        if st.session_state.messages:
            # Reverse the messages so the latest is on top
            for i, message in enumerate(reversed(st.session_state.messages)):
                if message["role"] == "user":
                    st.markdown(f"**You**: {message['content']}")
                else:
                    st.markdown(f"**CropGuard**: {message['content']}")
                
                # Add a separator line between conversations
                if i < len(st.session_state.messages) - 1:
                    st.markdown("---")

    elif selection == "Crop Recommendation":
        st.title("Crop Recommendation System")

        # Initialize variables with None or empty values
        temp = st.number_input("Enter the temperature (in Celsius):", value=None)
        hum = st.number_input("Enter the humidity (in percentage):", value=None)
        ph = st.number_input("Enter the pH value of the soil:", value=None)
        water = st.number_input("Enter water availability (in liters/m):", value=None)
        season = st.selectbox("Select the season:", [''] + list(label_encoder_season.classes_))  # Include an empty option

        if st.button("Recommend Crop"):
            # Validate inputs
            if temp is None or hum is None or ph is None or water is None or season == '':
                st.error("Please provide values for all fields before requesting a recommendation.")
            else:
                # Make the crop recommendation if all inputs are provided
                recommended_crop = predict_crop(temp, hum, ph, water, season)
                st.write(f"The recommended crop is: **{recommended_crop}**")


if __name__ == "__main__":
    main()
