import torch
import gradio as gr
from PIL import Image
from torchvision import transforms
from transformers import ViTForImageClassification, ViTImageProcessor
import numpy as np

# Load the pre-trained ViT model and processor
image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

# Define the image transformation pipeline for the test set
test_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
])

# Load your trained model (originally trained with 22 classes)
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224-in21k",  # Pretrained model from HuggingFace
    num_labels=22,  # Load with the original number of labels (22)
)

# Modify the classifier to have the correct number of labels (4)
model.classifier = torch.nn.Linear(model.classifier.in_features, 4)

# Set up the new id2label and label2id mappings for the 4-class classification
id2label = {0: 'Weed', 1: 'Pest', 2: 'Infected', 3: 'Diseased'}
label2id = {'Weed': 0, 'Pest': 1, 'Infected': 2, 'Diseased': 3}

model.id2label = id2label
model.label2id = label2id

# Set the model to evaluation mode
model.eval()

# Define the function to classify the image
def classify_image(image: Image.Image):
    # Preprocess the image
    image = test_transforms(image).unsqueeze(0).to(model.device)

    # Perform inference
    with torch.no_grad():
        outputs = model(image)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=-1).item()

    # Convert the predicted class index to the corresponding label
    predicted_label = id2label[predicted_class]
    return predicted_label

# Gradio interface setup
iface = gr.Interface(
    fn=classify_image,  # Function to classify the image
    inputs=gr.Image(type="pil", label="Upload Image"),  # Input type is an image
    outputs=gr.Label(num_top_classes=1, label="Predicted Label"),  # Output is the predicted label
    live=True,  # Optional live prediction
    title="Pest and Weed Detection",  # Title of the app
    description="Upload an image of a leaf to classify whether it's a pest, weed, infected, or diseased."  # Description
)

# Launch the Gradio interface
iface.launch()
