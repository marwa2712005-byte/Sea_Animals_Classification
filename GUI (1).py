import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


CLASSES = [
    'Clams', 'Corals', 'Crabs', 'Dolphin', 'Eel', 'Fish', 
    'Jelly Fish', 'Lobster', 'Nudibranchs', 'Octopus', 'Otter', 
    'Penguin', 'Puffers', 'Sea Rays', 'Sea Urchins', 'Seahorse', 
    'Seal', 'Sharks', 'Shrimp', 'Squid', 'Starfish', 
    'Turtle_Tortoise', 'Whale'
]


@st.cache_resource
def load_model():
    model = models.resnet18(pretrained=False)
    num_classes = len(CLASSES)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load('resnet18_animals.pth', map_location='cpu'))
    model.eval()
    return model


def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return transform(image).unsqueeze(0)


def predict(image, model):
    img_tensor = preprocess_image(image)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)
    
    predicted_class = CLASSES[predicted_idx.item()]
    confidence_score = confidence.item() * 100
    
    return predicted_class, confidence_score


def main():
    
    st.set_page_config(
        page_title="Sea animal classification ",
        page_icon="🐠",
        layout="centered"
    )
    
    
    st.markdown("""
        <style>
        .main {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        .stApp {
            background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        }
        h1 {
            color: #1565c0;
            text-align: center;
            font-family: 'Arial Black', sans-serif;
            padding: 20px;
            background: white;
            border-radius: 15px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .upload-text {
            font-size: 24px;
            color: #0d47a1;
            text-align: center;
            font-weight: bold;
            padding: 20px;
            background: white;
            border-radius: 10px;
            margin: 20px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .result-box {
            padding: 30px;
            background: linear-gradient(135deg, #42a5f5 0%, #1e88e5 100%);
            border-radius: 15px;
            text-align: center;
            color: white;
            box-shadow: 0 8px 16px rgba(0,0,0,0.2);
            margin: 20px 0;
        }
        .animal-name {
            font-size: 48px;
            font-weight: bold;
            margin: 20px 0;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .confidence {
            font-size: 32px;
            color: #ffeb3b;
            font-weight: bold;
        }
        </style>
    """, unsafe_allow_html=True)
    
    
    st.markdown("<h1>sea animal classification </h1>", unsafe_allow_html=True)
    
    
    st.markdown("""
        <div class='upload-text'>
            👇 choose an image 👇
        </div>
    """, unsafe_allow_html=True)
    
    
    st.write("")
    
    
    uploaded_file = st.file_uploader(
        "click here to choose an image",
        type=['jpg', 'jpeg', 'png'],
        help="choose JPG or PNG img"
    )
    
    
    if uploaded_file is not None:
        
        image = Image.open(uploaded_file).convert('RGB')
        
        
        col1, col2, col3 = st.columns([1, 3, 1])
        
        with col2:
            
            st.image(
                image, 
                caption='your img ',
                use_column_width=True,
                output_format='PNG'
            )
            
            st.write("")
            
            if st.button('🔍 find animal type ', use_container_width=True):
                with st.spinner(' please wait'):
                    
                    model = load_model()
                    predicted_class, confidence = predict(image, model)
                
                
                st.markdown(f"""
                    <div class='result-box'>
                        <div style='font-size: 24px; margin-bottom: 10px;'>
                            ✨ result ✨
                        </div>
                        <div class='animal-name'>
                            🐚 {predicted_class} 🐚
                        </div>
                        <div class='confidence'>
                            accuracy : {confidence:.1f}%
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
                
                if confidence > 80:
                   
                    st.success("high accuracy")
                elif confidence > 70:
                    st.info("accepted accuracy")
                else:
                    st.warning("not accurate")
    
    
    st.write("")
    st.write("")
    with st.expander("available animal to classify"):
        cols = st.columns(3)
        for idx, animal in enumerate(CLASSES):
            with cols[idx % 3]:
                st.write(f"🔹 {animal}")

if __name__ == "__main__":
    main()