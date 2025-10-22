# 🎭 Facial Emotion Recognition - Capstone Project
[![Buy Me A Coffee](https://img.shields.io/open-vsx/stars/redhat/java?color=D8B024&label=buy%20me%20a%20coffee&style=flat)](https://www.buymeacoffee.com/utsanjan)‎ ‎
[![](https://dcbadge.limes.pink/api/server/uavTPkr?style=flat)](https://discord.gg/ZuuWJm7MR3)‎ ‎ 
[![](https://img.shields.io/github/license/utsanjan/Emotion-Recognition?logoColor=red&style=flat)](https://github.com/utsanjan/Emotion-Recognition/blob/main/LICENSE)‎ ‎
[![](https://img.shields.io/github/languages/count/utsanjan/Emotion-Recognition?style=flat)](https://github.com/utsanjan/Emotion-Recognition/search?l=shell)‎ ‎
[![](https://img.shields.io/github/languages/top/utsanjan/Emotion-Recognition?color=light%20green&style=flat)](https://github.com/utsanjan/Emotion-Recognition)‎ ‎

This project provides a comprehensive end-to-end solution for detecting and classifying human emotions - such as happiness, sadness, and anger - based on facial expressions. By leveraging deep learning and computer vision techniques, this repository delivers a state-of-the-art implementation of Facial Emotion Recognition (FER). This repository implements **Facial Emotion Recognition (FER)** using:
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

## 📂 Project Structure
```
FER-Capstone/
│
├── data/
│   ├── fer2013.csv              # Original dataset (from Kaggle)
│   ├── cropped_faces/           # Preprocessed dataset
│   │   ├── train/<class>/
│   │   └── val/<class>/
│   └── sample_images/           # Optional test images
│
├── models/
│   ├── baseline_cnn.h5
│   └── mobilenet_emotion.h5
│
├── outputs/
│   ├── figures/                 # Confusion matrices, plots
│   ├── generated_faces/         # DCGAN samples
│   └── generated_music/         # MIDI outputs
│
├── scripts/
│   ├── preprocess.py
│   ├── train_baseline.py
│   ├── train_transfer.py
│   ├── evaluate.py
│   ├── webcam_demo.py
│   ├── dcgan.py
│   └── emotion_to_midi.py
│
├── utils/
│   ├── mtcnn_face_crop.py
│   └── data_loader.py
│
├── notebooks/
│   ├── 01_preprocess.ipynb
│   ├── 02_train_baseline.ipynb
│   ├── 03_train_transfer.ipynb
│   ├── 04_evaluate.ipynb
│   └── 05_webcam_demo.ipynb
│
├── requirements.txt
└── README.md
```

## ⚙️ Installation & Setup

### 1️⃣ Clone the repo
```bash
git clone https://github.com/<your-username>/FER-Capstone.git
cd FER-Capstone
```

### 2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Download dataset
Download FER2013 from Kaggle:  
[https://www.kaggle.com/datasets/nicolejyt/facialexpressionrecognition](https://www.kaggle.com/datasets/nicolejyt/facialexpressionrecognition)

Place it in:
```bash
data/fer2013.csv
```

## 🧠 Preprocessing
Convert FER CSV to cropped, normalized images:
```bash
python scripts/preprocess.py --csv data/fer2013.csv --out data/cropped_faces --target-size 224 --use-mtcnn
```

## 🏋️‍♂️ Training

### Baseline CNN
```bash
python scripts/train_baseline.py --data data/cropped_faces --epochs 25 --batch 64
```

### Transfer Learning (MobileNetV2)
```bash
python scripts/train_transfer.py --data data/cropped_faces --arch mobilenet --epochs 30 --input-size 224
```

## 📊 Evaluation
Generate confusion matrix and classification report:
```bash
python scripts/evaluate.py --model models/mobilenet_emotion.h5 --data data/cropped_faces/val
```

## 🎥 Real-Time Demo
Run live webcam emotion recognition:
```bash
python scripts/webcam_demo.py --model models/mobilenet_emotion.h5 --data-dir data/cropped_faces --input-size 224
```
Press `q` to quit webcam window.

## 🧬 Optional - DCGAN (Face Generation)
Train GAN to generate facial expressions:
```bash
python scripts/dcgan.py --data data/cropped_faces/train --out outputs/generated_faces --epochs 20000
```

## 🎵 Optional - Emotion-to-Music
Generate MIDI melody for any emotion:
```bash
python scripts/emotion_to_midi.py --emotion Happy --out outputs/generated_music/happy.mid
```

## 🧾 Dataset References
- [FER2013 (Kaggle)](https://www.kaggle.com/datasets/nicolejyt/facialexpressionrecognition)
- [UTKFace Cropped Dataset (Hugging Face)](https://huggingface.co/datasets/UTKFace)

## 🧪 Tech Stack
- Python 3.10+
- TensorFlow / Keras
- OpenCV
- MTCNN
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

## 📜 License
This project is licensed under the MIT License - see LICENSE file for details.

## 📬 Contact
- Questions? Reach out on [Instagram](https://www.instagram.com/utsanjan/)
- Explore more on my [YouTube Channel](https://www.youtube.com/DopeSatan)
- Join the [Discord Community](https://discord.gg/ZuuWJm7MR3)