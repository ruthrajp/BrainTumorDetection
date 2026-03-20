from flask import Flask, render_template, request
import torch
from torch import nn
from torchvision import transforms
from torchvision.models import shufflenet_v2_x1_0
from PIL import Image

# Initialize the Flask application
app = Flask(__name__)

# Load the model
num_classes = 4  # Change this to the number of classes in your trained model
model = shufflenet_v2_x1_0(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load('./shufflenet_model.pth'))
model.eval()

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Define transforms for your dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Class names
class_names = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']

# Prediction Function
def predict(image_path):
    model.eval()
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output.data, 1)
    return class_names[predicted.item()]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_route():
    if 'file' not in request.files:
        return "No file uploaded", 400
    file = request.files['file']
    if file.filename == '':
        return "No file selected", 400

    image_path = f"./static/{file.filename}"
    file.save(image_path)

    predicted_class = predict(image_path)
    return render_template('index.html', prediction=predicted_class, image_file=file.filename)

if __name__ == '__main__':
    app.run(debug=True)
