# 🫁 Chest X-Ray Pneumonia Detection

This project is focused on detecting **Pneumonia** from **Chest X-Ray images** using different deep learning architectures: **CNN**, **ResNet**, **Graph Neural Networks (GNN)**, and a **Multimodal Deep Learning** approach inspired by recent research.

---

## 📁 Dataset

We use the **Chest X-Ray Pneumonia** dataset from Kaggle:  
🔗 [Chest X-Ray Images (Pneumonia) | Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

- The dataset includes labeled chest X-ray images:
  - **Pneumonia**
  - **Normal**
- It is **imbalanced**, with more pneumonia images than normal, so special care was taken during training (e.g., augmentation).

---

## 🧠 Models Implemented

### 1️⃣ CNN (Convolutional Neural Network)

**📈 Accuracy: ~92% (Test)**

**Highlights:**
- Implemented a custom CNN from scratch.
- Faced **overfitting** initially → overcame it with:
  - `Dropout`
  - `L2 Regularization`
  - `EarlyStopping`
- Achieved excellent performance in:
  - **Precision**
  - **Recall**
  - **F1 Score**

✅ CNN proved to be the **best performing model** in this project.

---

### 2️⃣ ResNet (Transfer Learning)

**📈 Accuracy: ~85% (Test)**

**Highlights:**
- Used **ResNet** with **ImageNet pretrained weights**.
- Experimented with:
  - **Freezing** and **unfreezing** layers
  - **Fine-tuning** to adapt to medical domain
- Still has room for improvement via:
  - Advanced **data augmentation**
  - Exploring **learning rate schedules** and **optimizers**

---

### 3️⃣ GNN (Graph Neural Network)

**Model: Graph Attention Network (GAT)**

**Highlights:**
- Used the repo: `Chest-X-Ray-Detection`
- Converted image data into graph-structured input for GAT
- Aimed to capture **relational dependencies** between image patches
- Interesting experiment showing how graph-based reasoning can be applied in medical imaging

---

### 4️⃣ Multimodal Approach (Inspired by Research Paper)

🔗 [Multimodal CNN - Kaggle Notebook](https://www.kaggle.com/code/rishitkar/multimodalcnn)

**Highlights:**
- Inspired by **cross-modal contrastive learning** techniques
- Extracted:
  - **Scaled WSI (Whole Slide Image)**
  - **Image Patches**
- Used **parallel encoders** to extract features
- Applied a **contrastive module** for multimodal fusion

🖼️ Multimodal Architecture Example:
![Multimodal Architecture](Screenshot%202025-06-22%20163227.png)

---

## 🛠️ Technologies Used

- Python, NumPy, Pandas
- PyTorch, TensorFlow/Keras
- Torch Geometric (for GNN)
- OpenCV, Matplotlib
- Kaggle Kernels for training and evaluation

---

## ✅ Results Summary

| Model         | Accuracy | Highlights                                  |
|---------------|----------|---------------------------------------------|
| **CNN**       | **92%**  | Best model, overcame overfitting            |
| **ResNet**    | 85%      | Transfer learning with ImageNet weights     |
| **GNN (GAT)** | ~        | Graph-based patch learning                  |
| **Multimodal**| ~        | Combined global & patch views with contrastive learning |

---

## 🚀 Future Work

- Improve ResNet with better augmentation, schedulers, and optimizers
- Apply **self-supervised** or **semi-supervised** learning
- Optimize **GNN** and **Multimodal** designs for deployment

---

## 👨‍💻 Author

**Rishit Kar**  
📧 [LinkedIn](https://www.linkedin.com/in/rishitkar/)  
🧠 AI | 💻 ML | 📊 Research | 🔬 Deep Learning

