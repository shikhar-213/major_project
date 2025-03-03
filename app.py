from flask import Flask, request, jsonify, render_template, send_from_directory
import torch
from torchvision import transforms
from PIL import Image
import os
import io
import uuid  # To generate unique image filenames

app = Flask(__name__, static_folder="static", template_folder="templates")

# Load the trained model
model = torch.jit.load("plant_disease_cnn_scripted.pt")  # Use the scripted model
model.eval()

# **Get class names from dataset folders**
DATASET_PATH = "plant_data"  # Update if needed
class_names = sorted(os.listdir(DATASET_PATH))  # Folder names as class labels

# **Define upload folder**
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Create folder if not exists

# Define image preprocessing
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# **Serve the HTML Page**
@app.route("/")
def home():
    return render_template("index.html")

# **Serve Uploaded Images**
@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

# **Prediction Route**
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    
    # **Save uploaded image**
    filename = str(uuid.uuid4()) + ".jpg"  # Unique filename
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    # **Process Image for Prediction**
    image = Image.open(file_path)
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image)
        predicted_class_index = torch.argmax(output, 1).item()
        predicted_class_name = class_names[predicted_class_index]  # Get disease name

    # **Return prediction & image URL**
    return jsonify({
        "predicted_class": predicted_class_name,
        "image_url": f"/uploads/{filename}"
    })

if __name__ == '__main__':
    app.run(debug=True)