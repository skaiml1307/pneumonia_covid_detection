# Pneumonia & COVID-19 Detection from Chest X-rays

A deep learning project for classifying **Normal**, **COVID-19**, and **Pneumonia** chest X-ray images.  
Built with **TensorFlow/Keras**, tested on the [Mendeley COVID19, Pneumonia and Normal Chest X-ray PA Dataset].


##  Overview
- **Baseline CNN** → trained from scratch, achieved ~90% accuracy.  
- **ResNet50 Transfer Learning** → fine-tuned on ImageNet weights, achieved ~92% accuracy.  
- **ResNet101 (discarded)** → reached only ~42% accuracy, showing overfitting and poor generalization.  


### Setup & Installation
git clone https://github.com/skaiml1307/pneumonia_covid_detection.git

cd pneumonia_covid_detection

pip install -r requirements.txt


#### Dataset Preparation
python src/prepare_data.py


##### Training Models

- Baseline CNN (scratch training)
python src/model_cnn.py

- ResNet50 Transfer Learning
python src/resnet50.py

- ResNet101 (discarded)
Tried deeper architecture → only ~42% accuracy
Overfitting observed, excluded from final pipeline


###### Results
"Baseline CNN → 90.09%"
"ResNet50 Transfer Learning → 92.4%"
"ResNet101 → discarded (~42% accuracy)"


###### Key Learnings
"Not all deeper architectures guarantee better accuracy (ResNet101 failed)."
"EarlyStopping prevents overfitting and stabilizes results."
"Proper dataset stratification ensures balanced class representation."


###### How to Reproduce
python src/prepare_data.py
python src/model_cnn.py
python src/resnet50.py

ls results/


#  License
"This project is for educational and research purposes."
