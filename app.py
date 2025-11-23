import torch
import gradio as gr
from PIL import Image
from torchvision import transforms
from transformers import ViTForImageClassification, ViTImageProcessor
import numpy as np

# Assuming you have the model and data already loaded, including your dataset and model setup
# If you're loading from disk or have already trained, make sure to load your model appropriately

# Define transforms
image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
test_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
])

# Load the trained model (make sure the model is already trained and saved)
model = ViTForImageClassification.from_pretrained(
    "./vit-weed-pest-model",  # path to saved model
    num_labels=4,  # Adjust according to your number of labels
    id2label={0: 'Weed', 1: 'Pest', 2: 'Infected', 3: 'Diseased'},
    label2id={'Weed': 0, 'Pest': 1, 'Infected': 2, 'Diseased': 3}
)

model.eval()  # Set the model to evaluation mode

# Define the function to classify the image
def classify_image(image: Image.Image):
    # Preprocess the image to match the format used during training
    image = test_transforms(image).unsqueeze(0).to(model.device)

    # Perform inference
    with torch.no_grad():
        outputs = model(image)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=-1).item()

    # Convert the predicted class index to the corresponding label
    id2label = {0: 'Weed', 1: 'Pest', 2: 'Infected', 3: 'Diseased'}  # You should already have this from your training
    predicted_label = id2label[predicted_class]
    return predicted_label

# Gradio Interface
iface = gr.Interface(
    fn=classify_image,  # Function to classify the image
    inputs=gr.inputs.Image(type="pil", label="Upload Image"),  # Input image
    outputs=gr.outputs.Label(num_top_classes=1, label="Predicted Label"),  # Output label
    live=True,  # Optional live prediction
    title="Pest and Weed Detection",  # Title of the app
    description="Upload an image of a leaf to classify whether it's a pest, weed, infected, or diseased."  # Description
)

# Launch the Gradio interface
iface.launch()
