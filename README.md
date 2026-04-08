# 🍽️ MetaboLens  
### Personalized Meal Recommendation System using Depth-Aware Volumetric Food Analysis for Multi-Morbidity Nutritional Management

---

## 📌 Overview  
MetaboLens is an AI-powered system designed to analyze food images and generate **personalized dietary recommendations** for individuals managing **diabetes and cholesterol conditions**.  

The system uses **deep learning techniques** to detect food items, estimate portion sizes, calculate nutritional values, and provide health-aware recommendations — all from a single image.

---

## 🎯 Objectives  
- Automatically recognize food items from images  
- Estimate portion sizes using depth-aware techniques  
- Calculate nutritional values from detected foods  
- Generate personalized meal recommendations  
- Support users with diabetes and cholesterol management  

---

## 🧠 System Pipeline  

1. Image Upload  
2. Food Segmentation  
3. Food Classification *(hidden from user)*  
4. Portion Size Estimation  
5. Nutritional Analysis  
6. Personalized Recommendation Generation  

---

## 🏗️ Technologies Used  

### Programming Language  
- Python  

### Machine Learning / AI  
- PyTorch  
- TensorFlow  
- EfficientNet-B2 (Classification)  
- MobileNet + DeepLabV3 (Segmentation)  

### Libraries  
- OpenCV  
- Albumentations  
- NumPy  
- Pandas  
- Scikit-learn  

### Frontend  
- Streamlit  

### Development Tools  
- Visual Studio Code  
- Jupyter Notebook / Kaggle  

### Version Control  
- Git & GitHub  

---

## 📊 Features
- 📷 Upload food images
- 🍕 Automatic food detection
- ⚖️ Portion size estimation
- 🧮 Nutritional value calculation
- ❤️ Health-based recommendations
- 🧑‍⚕️ Designed for diabetes and cholesterol patients

---

## 📈 Evaluation Metrics

### Classification
- Accuracy
- Precision
- Recall
- F1 Score

### Segmentation
- Pixel Accuracy
- Mean IoU
- Dice Score

---

## ⚠️ Limitations
- Depth estimation is approximated because no real depth sensor was used
- Performance may vary with complex or mixed meals
- Limited dataset diversity may affect generalization
- No authentication system was implemented in the final prototype

---

## 🔮 Future Enhancements
- Add user authentication and role-based access
- Improve depth estimation using real-world calibration
- Expand the food dataset for better accuracy
- Develop a mobile application version
- Integrate with real-time nutrition databases

---

## 🎥 Demo Video
▶ Watch the Project Demo -->  https://youtu.be/9efkFH5svGs

---

## 👩‍💻 Author
**Kavindi Ranasingha**  
Final Year Undergraduate – Computer Science  

---

## 📜 License
This project is intended for academic and research purposes only. No commercial use is permitted.
