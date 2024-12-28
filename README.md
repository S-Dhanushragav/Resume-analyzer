# **AI-Powered Resume Analyzer**  

An innovative machine learning project designed to analyze resumes, predict their category, and assign confidence scores. This project uses advanced natural language processing (NLP) techniques to preprocess resume data, extract insights, and evaluate resumes based on domain-specific expertise.

---

## **Project Overview**  
This project aims to build a resume analysis system using a machine learning model trained on resume data. It supports PDF uploads, extracts text content, and categorizes resumes into predefined categories such as HR, IT, Healthcare, etc., while providing a confidence score for the prediction.  

### **Key Features**  
- Extracts and preprocesses text from resume PDF files.  
- Predicts the job category of the resume using a trained model.  
- Assigns a confidence score to the prediction.  
- Interactive UI for uploading resumes and visualizing predictions (optional Streamlit app).  

---

## **Dataset**  
Dataset link: https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset
The dataset was sourced from [Kaggle](https://www.kaggle.com/). It contains:  
- **2400+ Resumes** in string and PDF format.  
- Labels: Categories such as HR, IT, Finance, etc.  
- CSV file containing extracted text, HTML content, and category labels.  
- Folders with resumes categorized by domain.  

### **Data Description**  
- **ID**: Unique identifier for each resume.  
- **Resume_str**: Extracted text content of the resume.  
- **Resume_html**: HTML content of the resume as scraped from the web.  
- **Category**: Job domain the resume belongs to.  

---

## **Technologies Used**  
### **Programming Language:**  
- Python  

### **Libraries:**  
- **Data Handling:** pandas, numpy  
- **Text Preprocessing:** nltk, re  
- **Machine Learning:** scikit-learn, xgboost  
- **PDF Processing:** pdfplumber  
- **Model Deployment (optional):** Streamlit  

---

## **Workflow**  

### **1. Data Preprocessing**  
- Cleaned text data using NLP techniques (stopword removal, lemmatization, etc.).  
- Converted text to numerical features using **TF-IDF Vectorization**.  

### **2. Model Training**  
- Trained a **Random Forest Classifier** for predicting resume categories.  
- Achieved a model accuracy of **76%** after tuning.  

### **3. PDF Evaluation**  
- Extracted text from uploaded PDFs using `pdfplumber`.  
- Preprocessed the extracted text and transformed it using the trained **TF-IDF vectorizer**.  
- Predicted the category of the resume and provided a confidence score.  

---
