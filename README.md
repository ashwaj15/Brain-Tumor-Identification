
# 🧠 Brain Tumor Identification and Segmentation using MRI Images

This project focuses on identifying and segmenting **brain tumors from MRI scans** using **image processing** and **machine learning techniques**.  
By combining **Bilateral Filtering**, **Binary Thresholding**, **K-Means Clustering**, and **Convolutional Neural Networks (CNN)**, this model accurately detects and highlights tumor regions, providing a reliable tool for early diagnosis and research assistance.

---

## 🧩 Abstract
Brain tumors occur due to the abnormal growth of cells in the brain, and their progression can lead to brain cancer.  
Medical imaging techniques such as **CT scans**, **X-rays**, and **MRI** are used for tumor detection, among which **MRI** is considered the most reliable due to its ability to detect even minute structural abnormalities.

In this project:
- **Bilateral Filtering (BF)** is used for noise removal and image enhancement.
- **Binary Thresholding** and **K-Means Clustering** are used for tumor segmentation.
- **Convolutional Neural Network (CNN)** assists in reliable detection and prediction of tumor regions.

The model aims to automatically predict whether the subject has a tumor or not, reducing dependency on manual diagnosis and increasing the accuracy of detection.

---

## 💡 Introduction
According to **The International Association of Cancer Registration (IARC)**, over **28,000** brain tumor cases are reported in India each year, with more than **24,000** deaths annually due to lack of pre-diagnosis.

**Brain Tumor Segmentation** involves distinguishing tumor tissue from normal brain tissue in MRI scans.  
This project helps doctors and researchers visualize and identify tumor regions efficiently using image segmentation techniques.

---

## ⚙️ Existing Methods
Earlier approaches used **pre-trained deep learning models** such as:
- VGG-16  
- VGG-19  
- Inception V3  
- MobileNet V2  

While effective, these models require large datasets and computational resources.  
They also may not generalize well for all MRI formats or low-quality images.

<p align="center">
  <img src="images/existing_method.png" alt="Existing Methods Flow" width="600">
</p>

---

## 🚀 Proposed Model
Our proposed system simplifies the process by using:
1. **Input MRI Images**
2. **Pre-processing with Bilateral Filtering** to remove noise  
3. **Binary Thresholding** to find region of interest  
4. **K-Means Clustering** and **CNN-based segmentation** for accurate tumor detection  

✅ **Advantages:**
- Reduces computational complexity  
- Handles noise efficiently  
- Provides clear segmented tumor regions  
- Works effectively even on limited datasets  

<p align="center">
  <img src="images/proposed_model.png" alt="Proposed Model Flow" width="600">
</p>

---

## 🖥️ System Requirements

### 🔸 Software:
- **Programming Language:** Python  
- **Algorithm:** K-Means Clustering + CNN  
- **Libraries:** OpenCV, TensorFlow / PyTorch, NumPy, Pandas, Keras  
- **IDE:** Jupyter Notebook / Google Colab

### 🔸 Hardware:
- **Processor:** Intel Core i5 or higher  
- **RAM:** 4 GB or more  
- **Storage:** 10 GB free space  
- **Operating System:** Windows / macOS / Linux  
- **Display:** 1024×768 resolution or higher  

---

## 🧠 Technologies Used
| Technology | Purpose |
|-------------|----------|
| **Python** | Programming language for implementation |
| **OpenCV** | Image processing and filtering |
| **NumPy & Pandas** | Numerical operations and data handling |
| **Keras / TensorFlow** | Neural network creation and training |
| **Matplotlib / Seaborn** | Visualization of MRI outputs and segmentation |

---

## 🔄 Workflow

1. **Input** MRI image  
2. **Preprocessing**: Apply bilateral filtering to remove noise  
3. **Segmentation**: Use binary thresholding to find region of interest  
4. **Clustering**: Apply K-Means to identify tumor area  
5. **Detection**: CNN model predicts tumor presence  
6. **Output**: Segmented image showing tumor region

<p align="center">
  <img src="images/workflow.png" alt="Workflow Diagram" width="700">
</p>

---

## 🧾 Results
- Successfully detects tumor and non-tumor MRI images.
- Produces segmented masks highlighting tumor regions.
- Reduces false detections compared to threshold-only methods.
- Demonstrates efficiency on low-resource hardware.

### 📸 Sample Outputs:
<p align="center">
  <img src="images/input_mri.png" alt="Input MRI Image" width="300">
  <img src="images/segmented_output.png" alt="Segmented Output" width="300">
</p>

*(Upload your MRI input and output screenshots into an `/images` folder in your repo and rename them accordingly.)*

---

## 🔮 Future Scope
- Integrating **3D MRI image analysis** for volumetric segmentation.  
- Building a **web or mobile interface** for easy clinical access.  
- Implementing **U-Net or ResNet-based CNNs** for higher accuracy.  
- Expanding dataset diversity for improved generalization.

---

## 🧠 “Early detection saves lives — Empowering healthcare through machine learning.”


We used Convolutional Neural Network to predict whether the subject has Brain Tumor or not from MRI Images. Methods that could be used to increase accuracy includes, using large no. of images i.e., a larger dataset, Hyperparameter Tuning, Using a different Convolutional Neural Network Model may also result in higher accuracy.
