# Sea_Animals_Classification
Sea Animals Classification deep learning project using PyTorch and ResNet18 to classify 23 marine animal classes with data cleaning, augmentation, training, and evaluation.
# 🐬 Sea Animals Classification using ResNet18 (PyTorch)

## 📌 Project Overview
This project aims to classify sea animals into **23 different classes** using a deep learning model based on **ResNet18** with transfer learning.  
The dataset contains **13.7K images**, and several preprocessing steps such as balancing, augmentation, and normalization were applied to improve model performance.

---

## 🗂️ Dataset
- Total Images: 13.7K+
- Number of Classes: 23
- Dataset Loaded Using: `ImageFolder`
- Example Path Used in Project:

---

## 🧪 Data Processing Steps
### ✔️ Data Analysis
- Counted images per class
- Visualized class distribution using bar chart

### ✔️ Data Cleaning
- Balanced specific classes
- Removed extra images from over-represented classes

### ✔️ Data Augmentation
Applied:
- Resize → (224, 224)
- Random Horizontal Flip
- Rotation (15°)
- Color Jitter
- Normalization

---

## 🧠 Model Details
- Architecture: **ResNet18**
- Pretrained Weights: Yes
- Frozen Layers: All except:
  - `layer4`
  - `fully connected layer`
- Final Layer Replaced to match number of classes (23)
- Loss Function: `CrossEntropyLoss`
- Optimizer: `Adam`
- Learning Rate: `0.0001`

---

## 🏋️ Training
- Train / Validation split: 80% / 20%
- Epochs: 5
- Batch Size: 32
- Device: CPU/GPU depending on environment

---

## ✅ Evaluation
- Evaluated on validation set
- Metrics:
  - Training Loss
  - Validation Accuracy
  - Final Test Accuracy printed at the end

---

## 💾 Model Saving
After training, the model is saved as: resnet18_animals.pth

---

## ▶️ How to Run

### 1️⃣ Install Requirements as: pip install torch torchvision matplotlib numpy opencv-python


### 2️⃣ Prepare Dataset
Ensure dataset is structured like:
dataset images/
├── clams
├── sharks
├── crabs 
...

### 3️⃣ Run the Script
Just execute the Python file: SeaAnimalesClassification.ipynb  

---

## 📊 Visualization
The project visualizes:
- Number of images per class (before balancing)
- Number of images after balancing

---

## 📢 Notes
- Make sure dataset path is correct in the code
- You may adjust batch size, epochs, or learning rate for better performance
- Can upgrade to transfer learning with unfreezing more layers for higher accuracy

# 🐠 Sea Animal Classification GUI

## 📌 Code Overview
This project provides a **web-based GUI** for classifying sea animals using a **pretrained ResNet18 model**.  
Users can upload an image of a sea animal, and the app predicts the animal type along with a confidence score.  
The app is built using **Python**, **Streamlit**, and **PyTorch**.

---

## 🧠 Model
- Architecture: **ResNet18**  
- Number of Classes: 23 sea animals  
- Pretrained Weights: Loaded from `resnet18_animals.pth`  
- Prediction: Class name + confidence (%)  

**Classes Included: Clams, Corals, Crabs, Dolphin, Eel, Fish, Jelly Fish, Lobster, Nudibranchs, Octopus,
Otter, Penguin, Puffers, Sea Rays, Sea Urchins, Seahorse, Seal, Sharks, Shrimp, Squid,
Starfish, Turtle Tortoise, Whale


---

## ⚙️ Requirements
Install dependencies:
```bash
pip install streamlit torch torchvision pillow

---
##▶️ How to Run

1_ Clone the repository:
        git clone https://github.com/your-username/your-repo.git
        cd your-repo


2_ Make sure the model file resnet18_animals.pth is in the same directory

3_ Run the Streamlit app:
      streamlit run GUI(1).py


4_ Open the URL that appears in your browser (usually http://localhost:8501)


##🖼️ Usage:

    1_Upload an image (jpg, jpeg, or png)

    2_Click "find animal type"

    3_View the predicted class and confidence score

    4_Confidence Feedback:

    5_80% → High accuracy

    6_70–80% → Accepted accuracy

    7_<70% → Not accurate

##🎨 GUI Features:

      1_Responsive layout with 3-column design

      2_Stylish headers and result box

      3_Gradient background for main page

      4_Shows uploaded image preview

      5_Lists all available animals to classify in an expandable section

##💾 Files Needed:

     1_GUI.py → Streamlit GUI script

     2_resnet18_animals.pth → Trained PyTorch model weights

##📢 Notes:

       1_The app runs on CPU by default; can use GPU if available

       2_Ensure the uploaded images are RGB format

       3_You can update CLASSES or model weights if the dataset changes

---

## 👩‍💻 Team
- Mariem  
- Salma
- Raneem
- Shrouke
- Marwa

---

## 🎯 Goal
Achieve accuracy > 80% with balanced performance across all classes. 
