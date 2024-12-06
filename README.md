# YOLOv8 Burn Detection: Full Workflow Documentation

This document provides a comprehensive guide for setting up, training, and deploying a YOLOv8 model to detect the degree of burns (first, second, third degree). The workflow includes dataset organization, model training, testing, and deployment using a Flask-based web application.

---

## **1. Dataset Preparation**

### **1.1 Dataset Structure**
The dataset must be organized in the YOLO format, where each image has a corresponding `.txt` file containing annotations.

#### **Required Structure**:
```
dataset/
├── train/
│   ├── images/         # Training images
│   ├── labels/         # Training labels in YOLO format
├── val/
│   ├── images/         # Validation images
│   ├── labels/         # Validation labels in YOLO format
```

### **1.2 Organizing the Dataset**
If the images and labels are in the same folder:

1. Separate them into `images/` and `labels/` folders:
    ```python
    import os
    import shutil

    dataset_path = "path/to/dataset"
    images_folder = os.path.join(dataset_path, "images")
    labels_folder = os.path.join(dataset_path, "labels")

    os.makedirs(images_folder, exist_ok=True)
    os.makedirs(labels_folder, exist_ok=True)

    for file in os.listdir(dataset_path):
        if file.endswith(".jpg") or file.endswith(".png"):
            shutil.move(os.path.join(dataset_path, file), images_folder)
        elif file.endswith(".txt"):
            shutil.move(os.path.join(dataset_path, file), labels_folder)
    ```

2. Organize the dataset into `train/` and `val/` subsets.

### **1.3 YAML Configuration**
Create a `dataset.yaml` file to specify the dataset paths:

```yaml
train: dataset/train/images
val: dataset/val/images

test:  # Optional, if a test dataset exists

default: classes:
  names: ['first_degree', 'second_degree', 'third_degree']
```

---

## **2. Setting Up YOLOv8 Environment**

### **2.1 Install Required Libraries**
1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install the necessary libraries:
   ```bash
   pip install ultralytics flask
   ```

### **2.2 Verify Installation**
Test if YOLOv8 is installed correctly:
```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
print("YOLOv8 installed successfully")
```

---

## **3. Training the YOLOv8 Model**

### **3.1 Training Command**
Use the following command to train the YOLOv8 model:
```bash
from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # You can use yolov8n, yolov8m, etc.
model.train(data="dataset.yaml", epochs=50, imgsz=640, project="runs/train", name="burn_detection")
```

### **3.2 Outputs of Training**
After training, results will be saved in `runs/train/burn_detection/` and will include:
- `best.pt`: The best weights saved during training.
- Training plots for metrics like precision, recall, and mAP.

---

## **4. Testing the Model**

### **4.1 Running Inference**
Use the trained model for inference:
```python
from ultralytics import YOLO

model = YOLO("runs/train/burn_detection/weights/best.pt")
results = model.predict(source="path/to/test/image.jpg", save=True, save_txt=True)
print(results)
```
Results will be saved in `runs/detect/predictX`.

---

## **5. Deployment with Flask**

### **5.1 Project Structure**
The Flask web app should have the following structure:
```
flask_yolo_burn_detection/
├── app.py                 # Flask application script
├── static/
│   ├── uploads/           # Folder for uploaded images
│   ├── results/           # Optional folder for serving results
├── templates/
│   └── index.html         # HTML template
├── runs/
│   └── detect/            # YOLO inference results (default)
```

### **5.2 Flask Application Code**

#### **app.py**
```python
from flask import Flask, request, render_template, url_for, send_from_directory, jsonify
from ultralytics import YOLO
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)

# Set upload and result folders
UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "runs/detect"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# YOLO Model
model = YOLO("runs/train/burn_detection/weights/best.pt")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # File upload
        file = request.files["image"]
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)

            # Run YOLO inference
            results = model.predict(source=filepath, save=True)
            result_folder = results[0].save_dir
            result_image_path = os.path.join(result_folder, filename)

            # Construct URL
            result_image_url = url_for('serve_image', folder=os.path.basename(result_folder), filename=filename)
            return jsonify({"result_image_url": result_image_url})

    return render_template("index.html")

@app.route('/runs/detect/<folder>/<filename>')
def serve_image(folder, filename):
    return send_from_directory(os.path.join(RESULT_FOLDER, folder), filename)

if __name__ == "__main__":
    app.run(debug=True)
```

### **5.3 HTML Template**

#### **templates/index.html**
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>YOLOv8 Burn Detection</title>
</head>
<body>
    <h1>YOLOv8 Burn Detection</h1>
    <form method="POST" enctype="multipart/form-data">
        <label for="image">Choose an Image</label>
        <input type="file" name="image" id="image" required>
        <button type="submit">Upload and Detect</button>
    </form>
    <h2>Result:</h2>
    {% if result_image_url %}
        <img src="{{ result_image_url }}" alt="Result Image">
    {% endif %}
</body>
</html>
```

### **5.4 Running the Flask App**
Run the app:
```bash
python app.py
```
Visit `http://127.0.0.1:5000` in your browser to upload an image and view results.

---
