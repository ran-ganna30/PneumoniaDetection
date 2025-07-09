# 🫁 Pneumonia Detection from Chest X-rays using Deep Learning

This project uses deep learning (specifically a Convolutional Neural Network with transfer learning) to detect **pneumonia from chest X-ray images**. It utilizes the [Chest X-ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) dataset and is implemented in Python using the **FastAI v2** library.


---

## 📁 Dataset

* **Source:** [Kaggle - Chest X-ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
* **Structure:**

```
chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

* Contains 5,000+ chest X-ray images labeled as either **NORMAL** or **PNEUMONIA**.

---

## 🧠 Approach

### 🔧 Tools & Libraries:

* Python 3
* FastAI v2
* PyTorch (MPS backend on Mac / CPU fallback)
* Matplotlib, NumPy

### 🏗️ Model Architecture:

* Transfer learning using **ResNet18** (ResNet50 optional, but memory-intensive)
* Image size: 224×224
* Data augmentation applied via FastAI's `aug_transforms`
* Training on 80% of the `train` folder, with 20% used for validation

---

## 💡 Key Learnings

* How to implement image classification with transfer learning
* Importance of data augmentation in medical imaging
* Handling memory issues on Mac M1/M2 (MPS backend limitations)
* Model interpretability via `ClassificationInterpretation`

---

## 🚀 Future Improvements

* Use **Grad-CAM** to visualize model attention on X-rays
* Deploy via **Streamlit** or **Gradio**
* Extend to multiclass detection (e.g., distinguish viral vs. bacterial pneumonia)
* Integrate into a mobile/web app for remote diagnostics

---

## 🤝 Acknowledgements

* [Paul Mooney on Kaggle](https://www.kaggle.com/paultimothymooney)
* [FastAI](https://docs.fast.ai/)
* [The Clever Programmer (Aman Kharwal)](https://thecleverprogrammer.com/)

---

## 📬 Contact

Made by [Ranveer Ganna](mailto:rganna2@illinois.edu)
Feel free to connect with me for collaboration or feedback!
