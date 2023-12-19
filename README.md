# Network-Intrusion-Detection-System (NIDS)


Overview
This repository contains code for a Network Intrusion Detection System (NIDS) developed using Python and machine learning techniques. The system aims to detect and classify network intrusions or anomalies in a network environment.

Description
The Network Intrusion Detection System utilizes various machine learning algorithms to classify network traffic into normal and abnormal activities. The dataset used for training and testing consists of features extracted from network packets to identify potential intrusions.

Features
Data Preprocessing: Cleaning, transformation, and feature engineering techniques were applied to prepare the dataset for model training.
Machine Learning Models: Different classifiers such as Support Vector Machines (SVM), Random Forest, Naive Bayes, K-Nearest Neighbors (KNN), and Logistic Regression were implemented to predict and classify network intrusions.

Evaluation Metrics: The models were evaluated based on accuracy, precision, recall, F1-score, and ROC-AUC to assess their performance.

Contents
/data: Contains the dataset used for training and testing.
/models: Includes the trained machine learning models serialized for future use.
Network_Intrusion_Detection.ipynb: Jupyter Notebook containing the code implementation.
README.md: This file.

Instructions
Dataset: Ensure the dataset is available in the /data directory.
Setup: Install the required libraries by running pip install -r requirements.txt.
Execution: Run Network_Intrusion_Detection.ipynb using Jupyter Notebook or any compatible environment.

Requirements
Python 3.x
Jupyter Notebook
Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn]

Results
The README.md file provides detailed insights into the code implementation, model performance, and evaluation metrics.

