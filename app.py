import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
import pytorch_lightning as pl
from model import PneumoniaModel  

@st.cache_resource
def load_model():
    model = PneumoniaModel.load_from_checkpoint("model.ckpt")
    model.eval()
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return model


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.49], std=[0.248])
])

def preprocess(image):
    return transform(image).unsqueeze(0)

st.title(" Pneumonia Detection")
uploaded_file = st.file_uploader("Upload an X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded X-ray", use_column_width=True)

    model = load_model()
    input_tensor = preprocess(image).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.sigmoid(output).item()


    if prob >= 0.5:
        st.error(f" Pneumonia Detected (Confidence: {prob:.2%})")
    else:
        st.success(f" Normal (Confidence: {1 - prob:.2%})")
