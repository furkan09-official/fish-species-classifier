import streamlit as st
import joblib
import os
import pandas as pd
import numpy as np

# Load the trained model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

# Function to predict species and get corresponding image
def species_predict(length, weight, w_l_ratio):
    input_data = pd.DataFrame([[length, weight, w_l_ratio]], columns=['length', 'weight', 'w_l_ratio'])
    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)

    # Convert prediction from numpy array to scalar
    species = prediction.item()

    # Mapping species to image file names
    species_image_mapping = {
        'Setipinna taty': r'images/Setipinna taty.jfif',            
        'Anabas testudineus': r'images/Anabas testudineus.jpeg',      
        'Pethia conchonius': r'images/Pethia conchonius.jfif',        
        'Otolithoides biauritus': r'images/Otolithoides biauritus.jpg',   
        'Polynemus paradiseus': r'images/Polynemus paradiseus.jpg',     
        'Sillaginopsis panijus': r'images/Sillaginopsis panijus.jfif',    
        'Otolithoides pama': r'images/Otolithoides pama.jpg',        
        'Puntius lateristriga': r'images/Puntius lateristriga.jfif',     
        'Coilia dussumieri': r'images/Coilia dussumieri.webp'         
    }

    # Get the corresponding image path for the predicted species
    image_path = species_image_mapping.get(species, None)

    return species, image_path

# Streamlit UI
st.title('Fish Species Classifier')

length = st.number_input('Enter Length', min_value=0.0, step=0.1)
weight = st.number_input('Enter Weight', min_value=0.0, step=0.1)
w_l_ratio = st.number_input('Enter W_L_Ration', min_value=0.0, step=0.1)

if st.button('Predict'):
    species, image_path = species_predict(length, weight, w_l_ratio)

    # Display the predicted species
    st.write(f'The species is: {species}')

    # Display the corresponding image if it exists
    if image_path and os.path.exists(image_path):
        st.image(image_path, caption=species, use_column_width=True)
    else:
        st.write('No image is available for this species.')
