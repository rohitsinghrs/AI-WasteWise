<img width="1472" height="662" alt="Dashboard png" src="https://github.com/user-attachments/assets/99db0b24-97f0-47fc-a56b-52b3e75d245a" /># ♻️ AI-WasteWise — AI-Powered Waste Classification

<p align="center">
  <img src="<img width="683" height="192" alt="Banner png" src="https://github.com/user-attachments/assets/38bcd2ac-8c67-4b13-935f-f9b4b6a99c80" />
" alt="AI-WasteWise Project Banner" width="100%">
</p>

<p align="center">
  <strong>Making Waste Segregation Smarter with Artificial Intelligence</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow">
  <img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV">
</p>

<p align="center">
  An AI-powered web application that classifies waste images into different categories and helps users make more informed waste-disposal decisions.
</p>

<p align="center">
  <a href="https://github.com/rohitsinghrs/AI-WasteWise">View Repository</a>
  ·
  <a href="https://github.com/rohitsinghrs/AI-WasteWise/issues">Report a Bug</a>
</p>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Supported Waste Categories](#-supported-waste-categories)
- [Application Screenshots](#-application-screenshots)
- [How It Works](#-how-it-works)
- [Technology Stack](#-technology-stack)
- [Project Structure](#-project-structure)
- [Installation and Setup](#-installation-and-setup)
- [Usage](#-usage)
- [Model Information](#-model-information)
- [Future Improvements](#-future-improvements)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🌱 Overview

**AI-WasteWise** is a machine-learning-based waste classification application built with Python and Streamlit.

The application analyzes an uploaded image and predicts its waste category using a trained MobileNetV2-based deep learning model. It also provides prediction confidence and a dashboard for viewing prediction statistics and history.

The project aims to demonstrate how computer vision can support waste segregation and environmental awareness.

## ✨ Features

- 🖼️ **Image Upload:** Upload an image for waste classification.
- 🤖 **AI-Powered Prediction:** Classify images using a trained deep learning model.
- 📊 **Confidence Score:** Display the model's prediction confidence.
- 🔎 **Top Predictions:** Show the leading predicted categories, when available.
- 📈 **Analytics Dashboard:** View prediction statistics and category distribution.
- 🕒 **Prediction History:** Store and review previous predictions.
- 🗃️ **Database Integration:** Save prediction records for dashboard analytics.
- ♻️ **Waste Guidance:** Display category-related waste-disposal recommendations.
- 🛡️ **Input Validation:** Add checks to help prevent unsupported images from being accepted.

*Note: Model predictions are not guaranteed to be correct. Any non-waste rejection checks should be tested before being described as fully reliable.*

## ♻️ Supported Waste Categories

The model is trained to classify images into six categories:

| Category | Description |
|---|---|
| 📦 Cardboard | Cardboard boxes and packaging |
| 🫙 Glass | Glass bottles and containers |
| 🔩 Metal | Metal cans and other metal waste |
| 📄 Paper | Paper sheets and paper waste |
| 🧴 Plastic | Plastic bottles and packaging |
| 🗑️ Trash | Other waste represented in the training data |

The actual prediction quality depends on the training dataset, image quality, and model performance.

## 📸 Application Screenshots

### Dashboard

<!-- Replace dashboard.png with a real screenshot of your dashboard. -->
<p align="center">
  <img src="<img width="1472" height="662" alt="Dashboard png" src="https://github.com/user-attachments/assets/b35e20ed-5d7c-4ad2-a8e9-b13c643f60ed" />
" alt="AI-WasteWise Analytics Dashboard" width="90%">
</p>

### Waste Image Prediction

<p align="center">
  <img src="assets/prediction.png" alt="AI-WasteWise Waste Prediction" width="90%">
</p>

### Prediction History and Analytics

<p align="center">
  <img src=""D:\Downloads\Prediction.png.png"" alt="AI-WasteWise Prediction History" width="90%">
</p>

## ⚙️ How It Works

1. The user uploads a waste image through the Streamlit interface.
2. The application loads and preprocesses the image.
3. The trained MobileNetV2-based model generates predictions.
4. The application displays the predicted category and confidence score.
5. Prediction details are stored in the database for history and analytics.

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Application development |
| Streamlit | Interactive web interface |
| TensorFlow / Keras | Deep learning and model inference |
| MobileNetV2 | Image classification architecture |
| OpenCV | Image processing and optional face-detection checks |
| NumPy | Numerical operations |
| Pillow | Image loading and processing |
| Pandas | Data handling and analytics |
| SQLite or configured database | Prediction history and statistics |

## 📁 Project Structure

```text
AI-WasteWise/
│
├── app/
│   └── streamlit_app.py
│
├── src/
│   ├── predict.py
│   └── database.py
│
├── models/
│   └── wastewise_mobilenetv2.keras
│
├── assets/
│   ├── banner.png
│   ├── dashboard.png
│   ├── prediction.png
│   └── prediction-history.png
│
├── data/
├── requirements.txt
├── .gitignore
└── README.md
```

*Your local folder structure may contain additional files. Keep the structure here consistent with the files actually included in the repository.*

## 🚀 Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/rohitsinghrs/AI-WasteWise.git
cd AI-WasteWise
```

### 2. Create a virtual environment

**Windows PowerShell:**

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If OpenCV is required and is not already included in the dependencies:

```powershell
python -m pip install opencv-python-headless==4.10.0.84
```

### 4. Check model availability

Ensure the trained model is available at:

```text
models/wastewise_mobilenetv2.keras
```

If the model is not included in the repository, download or generate it according to the project's model setup instructions.

### 5. Run the application

```powershell
python -m streamlit run app/streamlit_app.py
```

Open the local URL displayed in your terminal, usually:

```text
http://localhost:8501
```

## 🧪 Usage

1. Launch the Streamlit application.
2. Upload an image using the image uploader.
3. Select the **Analyze Waste** button.
4. Review the predicted category and confidence score.
5. Explore the dashboard for category statistics and prediction history.

## 🧠 Model Information

- **Architecture:** MobileNetV2
- **Framework:** TensorFlow / Keras
- **Input image size:** 224 × 224 pixels
- **Output classes:** Cardboard, Glass, Metal, Paper, Plastic, and Trash
- **Model file:** `wastewise_mobilenetv2.keras`

The model predicts one of its trained categories. A high confidence score does not necessarily mean an image contains waste; reliable rejection of non-waste images requires appropriate validation or a separately trained detection mechanism.

## 🔮 Future Improvements

- Train a dedicated waste-versus-non-waste detector.
- Improve classification accuracy with a more diverse dataset.
- Add model evaluation metrics such as precision, recall, and F1-score.
- Improve the dashboard with clearer date-based analytics.
- Expand waste-disposal recommendations.
- Add automated testing and model monitoring.
- Improve deployment and documentation.

## 🤝 Contributing

Contributions, suggestions, and bug reports are welcome.

1. Fork the repository.
2. Create a feature branch.
3. Commit your changes.
4. Open a pull request describing your improvements.

For bugs or feature requests, visit the [Issues page](https://github.com/rohitsinghrs/AI-WasteWise/issues).

## 📄 License

Add a `LICENSE` file to specify how others may use, modify, and distribute this project. Until a license is added, do not assume that the repository is open for unrestricted reuse.

---

<p align="center">
  <strong>♻️ AI-WasteWise — Smarter Waste Segregation for a Cleaner Future</strong>
</p>
