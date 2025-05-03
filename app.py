import streamlit as st
from PIL import Image
from transformers import ViTFeatureExtractor, ViTModel
import torch
import torch.nn as nn


class TransformerReducer(nn.Module):
    def __init__(self, input_dim=768, output_dim=5, nhead=8, num_layers=2, dim_feedforward=1024):
        super(TransformerReducer, self).__init__()
        
        # Convert input to sequence with one token (reshape)
        self.input_proj = nn.Linear(input_dim, input_dim)
        
        # Positional encoding (not very useful for 1 token, but for consistency)
        self.positional_encoding = nn.Parameter(torch.zeros(1, 1, input_dim))
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection
        self.output_proj = nn.Linear(input_dim, output_dim)

    def forward(self, x):  # x: (batch_size, input_dim)
        x = self.input_proj(x).unsqueeze(1)  # Shape: (batch_size, 1, input_dim)
        x = x + self.positional_encoding
        x = self.transformer_encoder(x)  # Shape: (batch_size, 1, input_dim)
        x = x.squeeze(1)  # Remove sequence dim
        x = self.output_proj(x)  # Shape: (batch_size, output_dim)
        return x

# --- Load models ---
@st.cache_resource
def load_models():
    # Load ViT components
    feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
    vit = ViTModel.from_pretrained('google/vit-base-patch16-224')
    vit.eval()

    # Load your pretrained transformer model
    custom_transformer =  TransformerReducer(input_dim=768, output_dim=5, nhead=8, num_layers=2, dim_feedforward=1024)
    custom_transformer.load_state_dict(torch.load("transformer_model_weights.pth", map_location="cpu"))
    custom_transformer.eval()

    return feature_extractor, vit, custom_transformer

feature_extractor, vit_model, transformer_model = load_models()

# --- Preprocess uploaded image ---
def preprocess_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    inputs = feature_extractor(images=image, return_tensors="pt")
    return inputs["pixel_values"]

# --- Streamlit UI ---
# st.title("ViT + Custom Transformer Health Score")

# uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

#     with torch.no_grad():
#         # Preprocess
#         pixel_values = preprocess_image(uploaded_file)

#         # Step 1: ViT model
#         vit_output = vit_model(pixel_values=pixel_values)
#         cls_token = vit_output.last_hidden_state[:, 0]  # shape [1, 768]

#         # Step 2: Custom Transformer model
#         prediction = transformer_model(cls_token)
#         # health_score = torch.sigmoid(prediction).item() * 100  # Scale to [0, 100]
#     st.write("The value of my_variable is:", prediction[0])
#     # st.metric("Predicted Health Score", prediction)


st.title("ViT + Custom Transformer Health Score")

# Step 1: Capture or Upload Image
st.subheader("Step 1: Upload or Capture an Image")
use_camera = st.checkbox("Use Camera Instead")

if use_camera:
    uploaded_file = st.camera_input("Take a picture")
else:
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Input Image", use_column_width=True)

    with torch.no_grad():
        # Preprocess
        pixel_values = preprocess_image(uploaded_file)

        # Step 1: ViT model
        vit_output = vit_model(pixel_values=pixel_values)
        cls_token = vit_output.last_hidden_state[:, 0]  # shape [1, 768]

        # Step 2: Custom Transformer model
        prediction = transformer_model(cls_token)

    st.write("Predicted class logits:", prediction[0])
