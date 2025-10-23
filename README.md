# 🎭 Facial Emotion Recognition - Capstone Project
[![Buy Me A Coffee](https://img.shields.io/open-vsx/stars/redhat/java?color=D8B024&label=buy%20me%20a%20coffee&style=flat)](https://www.buymeacoffee.com/utsanjan)‎ ‎
[![](https://dcbadge.limes.pink/api/server/uavTPkr?style=flat)](https://discord.gg/ZuuWJm7MR3)‎ ‎ 
[![](https://img.shields.io/github/license/utsanjan/Emotion-Recognition?logoColor=red&style=flat)](https://github.com/utsanjan/Emotion-Recognition/blob/main/LICENSE)‎ ‎
[![](https://img.shields.io/github/languages/count/utsanjan/Emotion-Recognition?style=flat)](https://github.com/utsanjan/Emotion-Recognition/search?l=shell)‎ ‎
[![](https://img.shields.io/github/languages/top/utsanjan/Emotion-Recognition?color=light%20green&style=flat)](https://github.com/utsanjan/Emotion-Recognition)‎ ‎

This project provides a comprehensive solution for detecting and classifying human emotions - such as happiness, sadness, and anger - through facial expressions. Utilizing advanced deep learning and computer vision techniques, it features a state-of-the-art implementation of Facial Emotion Recognition (FER) through:
- **Convolutional Neural Networks (CNNs)** for baseline learning.
- **Transfer Learning** with **MobileNetV2** and **ResNet50** for improved accuracy.
- **MTCNN** for face detection & cropping.
- **DCGAN** for generating synthetic faces.
- Optional **MIDI music generation** based on predicted emotions.

## 🧩 Features
✅ Train baseline CNN on FER2013 dataset (48×48 grayscale).  
✅ Fine-tune transfer-learning model (MobileNetV2/ResNet50).  
✅ Real-time webcam emotion detection.  
✅ Evaluate model using confusion matrix & classification report.  
✅ Generate synthetic faces via DCGAN.  
✅ Create emotion-based melodies (MIDI).

## ⚙️ Installation & Setup

Follow these steps to set up the project on your local system:

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/utsanjan/Emotion-Recognition.git
cd Emotion-Recognition
```

### 2️⃣ Create and Activate a Virtual Environment
It’s best to isolate dependencies for this project.

**Windows (PowerShell):**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3️⃣ Install Required Dependencies
Make sure you have Python 3.8+ installed.

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

💡 **Note:** If you face issues with TensorFlow installation, install the correct version for your hardware:
```bash
pip install tensorflow==2.13.0  # CPU version
pip install tensorflow-macos==2.13.0  # for Apple Silicon
```

### 4️⃣ Download the FER Dataset
Use the FER-2013 dataset (or any compatible emotion dataset).

📦 **FER-2013 Dataset (Kaggle):**  
[https://www.kaggle.com/datasets/nicolejyt/facialexpressionrecognition](https://www.kaggle.com/datasets/nicolejyt/facialexpressionrecognition)

After downloading, place `fer2013.csv` in the `data/` folder:
```
project_root/
├── data/
│   └── fer2013.csv
```

### 5️⃣ Preprocess the Data
Run the preprocessing script to crop, clean, and organize images:
```bash
python scripts/preprocess.py --csv data/fer2013.csv --out data/cropped_faces --use-mtcnn
```

This will generate a folder structure like:
```
data/
├── cropped_faces/
│   ├── train/
│   ├── val/
│   └── test/
```

### 6️⃣ Train the Model
You can train either the baseline CNN or the transfer-learning model.

**Baseline CNN:**
```bash
python scripts/train_baseline.py
```

**Transfer Learning (e.g., MobileNet, VGG16, ResNet50):**
```bash
python scripts/train_transfer.py
```

Trained models are automatically saved in the `models/` directory:
```
models/
├── baseline_cnn.h5
├── mobilenet_emotion.h5
└── best_feature_extractor.keras
```

### 7️⃣ Evaluate the Model
Generate metrics and a confusion matrix:
```bash
python scripts/evaluate.py
```

### 8️⃣ Run the Real-Time Webcam Demo
Once your model is trained, launch real-time emotion detection:
```bash
python scripts/webcam_demo.py
```

Press `q` to exit the webcam window.

### 9️⃣ Optional - Emotion-to-Music
If you’ve enabled the `emotion_to_midi` module, your webcam demo can auto-generate MIDI files based on detected emotions.

Generated files are stored in:
```
outputs/midi/
```

## 🧾 Dataset References
- [FER2013 (Kaggle)](https://www.kaggle.com/datasets/nicolejyt/facialexpressionrecognition)
- [UTKFace Cropped Dataset (Hugging Face)](https://huggingface.co/datasets/UTKFace)

## 🧪 Tech Stack
- Python 3.10+
- OpenCV
- MTCNN
- TensorFlow / Keras
- Matplotlib / Seaborn
- PrettyMIDI for audio generation

## 📈 Results (Example)
| Model              | Accuracy       | Notes                              |
|--------------------|----------------|------------------------------------|
| Baseline CNN       | ~65%           | Simple grayscale CNN               |
| MobileNetV2 (TL)   | ~73–80%        | Transfer learning + 224px RGB      |
| ResNet50 (TL)      | ~78–82%        | Higher accuracy, more compute      |

## 🤝 Acknowledgements
- FER2013 dataset by Pierre-Luc Carrier & Aaron Courville
- Preprocessing logic inspired by [GSNCodes (GitHub)](https://github.com/GSNCodes)
- Keras DCGAN example - © Keras Team

## 📬 Contact
- Questions? Reach out on [Instagram](https://www.instagram.com/utsanjan/)
- Explore more on my [YouTube Channel](https://www.youtube.com/DopeSatan)
- Join the [Discord Community](https://discord.gg/ZuuWJm7MR3)
