🇹🇳🔍 Tunisian License Plate Recognition – Horizop Energy

Deep Learning system for detecting, extracting, and recognizing Tunisian vehicle license plates.
Developed during my engineering internship at Horizop Energy.

📌 Project Overview

This project implements a complete end-to-end License Plate Recognition (LPR) pipeline exclusively for Tunisian license plates.
It includes:

Plate detection using YOLO-based models

Plate extraction & segmentation

Character recognition (OCR) using a trained deep-learning model (ocrmodel.h5)

Data preprocessing, cleaning, and analysis through Jupyter notebooks

Character classification dataset for training OCR models

All stages of the system are documented and implemented inside organized Jupyter notebooks.

📁 Repository Structure
📦 Horizop-IA-License-Plate-Recognition
 ┣ 📁 Notebooks/
 ┃ ┣ Data Preprocessing & Cleaning - Horizop_version.ipynb
 ┃ ┣ Licence Plate Detection and Extraction - Horizop_version.ipynb
 ┃ ┣ Licence Plate Recognition - Horizop_version.ipynb
 ┃ ┣ Modeling for Licence Plate Recognition - Horizop_version.ipynb
 ┃ ┗ Main Script - Horizop_version.ipynb
 ┣ 📁 LP_extraction_test/
 ┃ ┗ Sample test images used for plate extraction
 ┣ 📁 Characters-Classification-Data/
 ┃ ┣ train/
 ┃ ┗ val/
 ┣ 📄 ocrmodel.h5 — Trained OCR model  
 ┣ 📄 darknet-yolov3.cfg — Detection model configuration  
 ┣ 📄 classes.names — YOLO classes for Tunisian plates  
 ┗ 📄 README.md

⭐ Key Features

🔍 License Plate Detection using YOLO (V3 architecture)

✂️ Plate Extraction & Segmentation

🔡 Deep Learning OCR with custom-trained character classifier

🧹 Dataset cleaning, augmentation, and preprocessing

📊 Multiple structured notebooks for transparency and reproducibility

🖼️ Real test images included for validation

🧠 Technologies Used

Python, OpenCV

TensorFlow / Keras

YOLOv3 (darknet-style config)

NumPy, Pandas

Scikit-learn

Matplotlib & Seaborn

Jupyter Notebook

🧪 Notebooks Explained
1️⃣ Data Preprocessing & Cleaning

Includes dataset filtering, augmentation, normalization, and annotation validation.

2️⃣ License Plate Detection & Extraction

Implements YOLO detection and automated cropping of plate regions.

3️⃣ Modeling for License Plate Recognition

Training, validation, metrics, and optimization of the OCR model.

4️⃣ License Plate Recognition

Applies the OCR classifier on segmented characters for full plate reconstruction.

5️⃣ Main Script

End-to-end pipeline combining:

Detection → Extraction → Segmentation → Recognition

📂 Dataset

Located in Characters-Classification-Data/train and Characters-Classification-Data/val.
Contains cleaned and labeled character images for OCR training.

🚀 How to Run
Install dependencies
pip install -r requirements.txt

Open notebooks
jupyter notebook

OR run the end-to-end script (if you create main.py)
python main.py

📈 Future Improvements

Move from YOLOv3 to YOLOv8/YOLOv10

Add real-time camera detection

Export the model to ONNX or TensorFlow Lite

Build a small web or mobile interface

📄 Internship Context

This work was developed during my engineering summer internship at Horizop Energy, focusing on real-world vehicle identification solutions for smart mobility and energy infrastructure.
