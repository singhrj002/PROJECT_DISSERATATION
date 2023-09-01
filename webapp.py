import streamlit as st
from PIL import Image
import torch
import torchvision
from torchvision.transforms import ToTensor
from torchvision.models import mobilenet_v2, resnet18
from torch import nn

# Class names for initial detection
initial_class_names = {
    0: "knot",
    1: "random",
    2: "rope"
}

# Class names for knot types and tensions
class_names = {
    0: "alpinebuttefly knot",
    1: "bowline",
    2: "clove_hitch",
    3: "fisherman's",
    4: "flemishbend",
    5: "overhandknot",
    6: "reefknot",
    7: "slipknot"
}

class_names_tension = {
    0: "loose",
    1: "tight",
    2: "very loose"
}

# Dictionary of video URLs for each knot type
video_links = {
       "alpinebuttefly knot": "https://www.youtube.com/watch?v=Nw9fNxSGqAo&t=53s",
    "bowline": "https://www.youtube.com/watch?v=Q9NqGd7464U",
    "clove_hitch": "https://www.youtube.com/watch?v=Gs9WyrzNjJs",
    "fisherman's": "https://www.youtube.com/watch?v=GhS3mhRsf-k",
    "flemishbend": "https://www.youtube.com/watch?v=oylzeGvO3Qc",
    "overhandknot": "https://www.youtube.com/results?search_query=overhand+knot",
    "reefknot": "https://www.youtube.com/watch?v=ZnZUCvqksFA",
    "slipknot": "https://www.youtube.com/watch?v=Yc4oLSHEiIQ"
}
# Initialize the model for knot type prediction
model1 = mobilenet_v2(pretrained=False)
model1.classifier[1] = nn.Linear(model1.last_channel, 8)
model1 = torch.load(r'C:\Users\singh\OneDrive\Desktop\project-knot\2_model1_full.pth', map_location=torch.device('cpu'))

# Preprocess image function
def preprocess_image(image):
    image = image.resize((224, 224))
    image = image.convert("RGB")
    return image

# Initialize the initial detection model
initial_model = mobilenet_v2(pretrained=False)
initial_model.classifier[1] = nn.Linear(initial_model.last_channel, 3)
initial_model = torch.load(r'C:\Users\singh\OneDrive\Desktop\project-knot\1_model3_full (1).pth', map_location=torch.device('cpu'))

# Initial detection function
def initial_predict(image):
    preprocessed_image = preprocess_image(image)
    input_tensor = ToTensor()(preprocessed_image)
    input_tensor = input_tensor.unsqueeze(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    initial_model.to(device)
    input_tensor = input_tensor.to(device)
    initial_model.eval()
    with torch.no_grad():
        output = initial_model(input_tensor)
    _, predicted_idx = torch.max(output, 1)
    return initial_class_names[predicted_idx.item()]

# Prediction function for knot type and tension
def predict_image(image):
    preprocessed_image = preprocess_image(image)
    input_tensor = ToTensor()(preprocessed_image)
    input_tensor = input_tensor.unsqueeze(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model1.to(device)
    input_tensor = input_tensor.to(device)
    model1.eval()
    with torch.no_grad():
        output = model1(input_tensor)
    _, predicted_idx = torch.max(output, 1)
    knot_type = class_names[predicted_idx.item()]

    # Depending on the predicted label, load the appropriate architecture
    if predicted_idx.item() in [3, 4, 6]:
        model2 = resnet18(pretrained=False)
        model2_path = f'C:\\Users\\singh\\OneDrive\\Desktop\\project-knot\\3.{predicted_idx.item() + 1}_resnet_model3_full.pth'
    else:
        model2 = mobilenet_v2(pretrained=False)
        model2_path = f'C:\\Users\\singh\\OneDrive\\Desktop\\project-knot\\3.{predicted_idx.item() + 1}_mobilenet_model3_full.pth'
    
    # Load the specific model weights for the predicted knot type
    model2.load_state_dict(torch.load(model2_path, map_location=torch.device('cpu')))
    
    model2.to(device)
    model2.eval()
    with torch.no_grad():
        tension_output = model2(input_tensor)
    _, tension_idx = torch.max(tension_output, 1)
    tension = class_names_tension[tension_idx.item()]

    return knot_type, tension


# Streamlit UI
logo = Image.open(r"C:/Users/singh/OneDrive/Desktop/project-knot/logo.jpg")
st.image(logo, width=150)

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    initial_prediction = initial_predict(image)
    
    if initial_prediction == "knot":
        if st.button("Predict"):
            predicted_knot, predicted_tension = predict_image(image)
            
            # Display the predicted knot and tension
            st.write("Predicted knot:", predicted_knot)
            st.write("Predicted tension:", predicted_tension)
            
            # Display the recommended video
            st.write("### Recommended Video")
            video_url = video_links.get(predicted_knot)
            if video_url:
                st.video(video_url)

        st.markdown("---")
        show_image = st.checkbox("Show Image")
        if show_image:
            st.write("### Image You Uploaded")
            st.image(image, use_column_width=True)
    else:
        st.write("No knot found.")

# Sidebar content
st.sidebar.write("This is a web application which can detect knot and tension in knot. Currently available for:")
st.sidebar.write("1. Alpinebutterfly knot")
st.sidebar.write("2. Bowline knot")
st.sidebar.write("3. Clove hitch knot")
st.sidebar.write("4. Fisherman knot")
st.sidebar.write("5. Flemish bend knot")
st.sidebar.write("6. Overhand knot")
st.sidebar.write("7. Reef knot")
st.sidebar.write("8. Slip knot")
st.sidebar.write("I extend my heartfelt gratitude to Dr. John Williamson for his invaluable contribution and guidance throughout this project.")
