import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import pickle
from torchvision import models
import torch.nn as nn

# Page config
st.set_page_config(page_title="Image Suggestion App", page_icon="🖼️")

st.title("🖼️ Image Suggestion App")
st.caption("Academic Project — Multi-Label Suggestions from Input Images")

class Phi3Model(nn.Module):
    def __init__(self, num_classes):
        super(Phi3Model, self).__init__()
        # Match training: we used resnet18 and replaced the final layer
        self.backbone = models.resnet18(weights=None)  # No pretrained weights needed
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)

@st.cache_resource
def load_model():
    # Load label index → suggestion text mapping
    with open("idx_to_suggestion.pkl", "rb") as f:
        idx_to_suggestion = pickle.load(f)

    num_classes = len(idx_to_suggestion)

    # Restore trained model architecture
    model = Phi3Model(num_classes)
    model.load_state_dict(torch.load("model.pth", map_location="cpu"))
    model.eval()
    return model, idx_to_suggestion

model, idx_to_suggestion = load_model()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def predict(image):
    image = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        outputs = model(image)
        probs = torch.sigmoid(outputs).squeeze(0)

        # Multi-label threshold classifier
        suggestions = [idx_to_suggestion[i] for i, p in enumerate(probs) if p > 0.5]

    return suggestions or ["No suggestion above confidence threshold"]

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    if st.button("🔍 Generate Suggestions"):
        with st.spinner("Analyzing image..."):
            result = predict(img)

        st.success("Suggestions:")
        for r in result:
            st.write("•", r)
