### 🚦 Traffic Light Recognition System

Welcome to the **Traffic Light Recognition System**, a deep learning-based project for detecting and recognizing traffic light states from images or video streams. The system is designed to classify traffic lights into states such as **🔴 Red**, **🟡 Yellow**, and **🟢 Green**, contributing to intelligent transportation and autonomous driving systems.

---

#### 📌 Project Overview

The objective of this project is to develop a model capable of accurately recognizing traffic light states in real-world environments.

##### 1️⃣ **Data preprocessing & augmentation**  
Prepare and improve training data through re-distribution.
- Traffic light images labeled by state:
  - 🔴 Red
  - 🟡 Yellow
  - 🟢 Green
  - ⚫ Off / Unknown


##### 2️⃣ **Model training**  
Train a YOLO deep learning model for **object detection** and **classification**.
- **Loss Function:** Cross-entropy / detection loss  
- **Optimizer:** Adam / SGD


##### 3️⃣ **Evaluation**  
Assess model performance on validation/test datasets.
- **Evaluation Metrics:** Accuracy, Precision, Recall


##### 4️⃣ **Results**  
Explore the generated graphs and result predictions through a **Streamlit interface**.


#### 🚀 Goal

Build a robust and accurate traffic light recognition system capable of operating under real-world traffic conditions.


---

#### GUI (`gui.py`)

A Streamlit-based interface for the dataset pipeline tools.

##### Run with:

```bash
pip install streamlit pandas
streamlit run gui.py
```

The sidebar selects between five sections:

##### 📊 BSTLD Stats
Shows class counts and null frame statistics for a BSTLD-formatted `.yaml` labels file.

##### 📈 YOLO Stats
Shows a per-split breakdown of class instance counts and null frames for a YOLO-formatted dataset directory.

##### 🔄 Convert BSTLD → YOLO
Converts a Bosch-format dataset to YOLO format, with options for image transfer method, color variant merging, and class ID offset.

##### ⚖️ Re-split Dataset
Merges and re-splits an existing YOLO dataset, either randomly or with stratified balancing to preserve class distribution across splits.

##### 🗂️ View Results
A lightweight file browser for navigating dataset directories. Displays `.png` / `.jpg` images inline and opens `.csv` files as interactive tables. Includes a breadcrumb trail for easy navigation.


