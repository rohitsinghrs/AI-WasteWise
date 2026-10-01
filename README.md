# \## Project Status

# 

# The core AI WasteWise pipeline is complete.

# 

# \### Completed

# 

# \- Dataset collection and organization

# \- Dataset exploratory analysis

# \- Train/validation/test split

# \- Data preprocessing and augmentation

# \- MobileNetV2 transfer learning model

# \- Class-weighted training for class imbalance

# \- Model evaluation

# \- Waste image prediction

# \- Disposal recommendation system

# \- Streamlit web application

# \- FastAPI backend

# \- API documentation through Swagger UI

# 

# \### Model Performance

# 

# The model was evaluated on 379 previously unseen test images.

# 

# \- Test Accuracy: 67.55%

# \- Number of Classes: 6

# \- Model: MobileNetV2

# \- Input Size: 224 × 224

# 

# \### Supported Waste Categories

# 

# 1\. Cardboard

# 2\. Glass

# 3\. Metal

# 4\. Paper

# 5\. Plastic

# 6\. Trash

# 

# \### Application Flow

# 

# User Image

# → Image Preprocessing

# → MobileNetV2

# → Waste Classification

# → Confidence Score

# → Disposal Recommendation

# 

# \### Current Limitation

# 

# The model is trained on the TrashNet dataset and therefore performance can vary on real-world images with different backgrounds, lighting conditions, object orientations, and waste categories.

# 

# Low-confidence predictions should be manually verified before disposal.

