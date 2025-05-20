import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
from PIL import Image
import torch
import CNN
import os
import torchvision.transforms.functional as TF
disease_info = pd.read_csv('disease_info.csv' , encoding='cp1252')
supplement_info = pd.read_csv('supplement_info.csv',encoding='cp1252')

model = CNN.CNN(39)    
model.load_state_dict(torch.load("plant_disease_model_1_latest.pt"))
model.eval()

def prediction(image_path):
    image = Image.open(image_path)
    image = image.resize((224, 224))
    input_data = TF.to_tensor(image)
    input_data = input_data.view((-1, 3, 224, 224))
    output = model(input_data)
    output = output.detach().numpy()
    index = np.argmax(output)
    return index
# Navigation menu for user dashboard

with st.sidebar:
    st.markdown(f"<h1 style='text-align: center; color: black;'><b>🏡Dashboard</b></h1>", unsafe_allow_html=True)
    st.image("https://www.pngkey.com/png/full/913-9135243_img-ploughing-indian-farmer-clipart.png",use_column_width=True)
    selected_tab = option_menu(
        menu_title=None,
        options=["Home", 'Disease Detection'],
    styles={
    "nav-link-selected": {"background-color": "green", "color": "white", "border-radius": "5px"},
    }
    )
if selected_tab=="Home":
    st.markdown('<h1 style="text-align: center; color: green;">Welcome to the Plant Disease Detection System</h1>', unsafe_allow_html=True)
    col1,col2,col3= st.columns([1, 4, 1])
    col2.image("https://content.us.lifeomic.com/sustainablenano/fcd02130-c47d-4b34-8ca4-c92d0952d241/5b586aa9-3791-4497-9bca-93258788e796/NanoAg-14.gif",use_column_width=True)
elif selected_tab == "Disease Detection":
    st.markdown(
        """
        <style>
        /* Apply background image to the main content area with transparency */
        .main {
            background-image: url('https://media.istockphoto.com/id/1250059761/photo/septoria-of-tomatoes-tomato-leaves-affected-by-septoria-lycopersici-fungus.jpg?s=612x612&w=0&k=20&c=GuJ1uqQjaGryWs8Qp1xmBOMSPdlUho8X8DKKkjYRV44=');
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-color: rgba(255, 255, 255, 0.6); /* Add a semi-transparent overlay */
            background-blend-mode: overlay; /* Blend the image with the overlay */
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown(
        """
        <div style="text-align: center; color: white;">
            <h1 style="color: maroon; font-size: 50px; text-align: center;">Crop Disease Detection</h1>
        </div>
        """,
        unsafe_allow_html=True
    )
    # File uploader
    image = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])
    if image:
        col1, col2, col3 = st.columns([3, 6, 1])
        col2.image(image, caption='Uploaded Image',width=250)

        # Save the uploaded image
        filename = image.name
        file_path = os.path.join('uploads', filename)
        with open(file_path, "wb") as f:
            f.write(image.getbuffer())
        try:
            # Perform prediction
            pred = prediction(file_path)

            # Fetch details based on prediction
            title = disease_info['disease_name'][pred]
            description = disease_info['description'][pred]
            prevent = disease_info['Possible Steps'][pred]
            image_url = disease_info['image_url'][pred]
            supplement_name = supplement_info['supplement name'][pred]
            supplement_image_url = supplement_info['supplement image'][pred]
            supplement_buy_link = supplement_info['buy link'][pred]
            if title!='Background Without Leaves':
                # Display results
                col1, col2, col3 = st.columns([2, 6, 1])
                col2.markdown(f"<h2 style='color:red;'>{title}</h2>", unsafe_allow_html=True)
                st.markdown(
                    f"""
                    <div style="text-align: justify; padding: 10px; background-color: #d3e876; border-radius: 20px; border: 1.5px solid black; margin-bottom: 20px;">
                        <h2 style="color: #111df7; font-size: 20px;"><b>Disease Description:</b></h2>
                        <p style="color: black; font-size: 15px;"><b>{description}</b></p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                st.write(f"")
                st.markdown(
                    f"""
                    <div style="text-align: justify; padding: 10px; background-color: #ffa1ef; border-radius: 20px; border: 1.5px solid black; margin-bottom: 20px;">
                        <h2 style="color: #111df7; font-size: 20px;"><b>Prevntion Steps:</b></h2>
                        <p style="color: black; font-size: 15px;"><b>{prevent}</b></p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                st.write(f"")
                st.markdown(
                    f"""
                    <div style="text-align: center; padding: 8px; background-color: #ffd5a1; border-radius: 30px; border: 1.5px solid black; margin-bottom: 10px;">
                        <h2 style="color: #111df7; font-size: 20px;"><b>Recommended Supplement:</b> {supplement_name}</h2>
                        <div style="text-align: center; margin-top: 10px;">
                            <img src="{supplement_image_url}" alt="Supplement Image" style="width: 300px; height: auto; border-radius: 15px; border: 1px solid black;">
                        </div>
                        <div style="margin-top: 15px;">
                            <a href="{supplement_buy_link}" target="_blank" style="text-decoration: none;">
                                <button style="background-color: red; color: white; font-size: 16px; padding: 10px 20px; border: none; border-radius: 10px; cursor: pointer;">
                                    Buy Supplement Here
                                </button>
                            </a>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(f"<h1 style='text-align: center; color:red;'>No Disease Detected</h1>", unsafe_allow_html=True)
        except:
            st.error('Invalid Image')
