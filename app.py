import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import pickle
from torchvision import models
import torch.nn as nn

st.set_page_config(page_title="Image Suggestion Model", page_icon="🖼️")

@st.cache_resource
def load_model():
    with open("idx_to_suggestion.pkl", "rb") as f:
        idx_to_suggestion = pickle.load(f)

    num_classes = len(idx_to_suggestion)

    model = models.resnet18(pretrained=False)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    model.load_state_dict(torch.load("model.pth", map_location="cpu"))
    model.eval()
    return model, idx_to_suggestion

model, idx_to_suggestion = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

st.title("🖼️ Image Suggestion App")
st.caption("Academic project — multi-label suggestions from images")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

def predict(image):
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image)
        probs = torch.sigmoid(outputs).squeeze(0)
        suggestions = [idx_to_suggestion[i] for i, p in enumerate(probs) if p > 0.5]
    return suggestions or ["No suggestion above threshold"]

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    if st.button("🔍 Get Suggestions"):
        result = predict(img)
        st.success("Suggestions:")
        for r in result:
            st.write("- ", r)
