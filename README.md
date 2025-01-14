
# Mandrill-Data Classification and Analysis
## Overview
This project focuses on the classification and analysis of textual data related to "Mandrill". The goal is to identify posts mentioning Mandrill and classify them correctly using machine learning techniques. The project involves the use of natural language processing (NLP) and machine learning models to classify posts as either "Mandrill" or "inne" (other).
The dataset is constructed from textual data and is processed using TF-IDF vectorization, followed by training machine learning models such as Naive Bayes and Logistic Regression. Additionally, performance evaluation is conducted through metrics such as accuracy, confusion matrices, and ROC-AUC curves.
![mandrill_thumb_3x4](https://github.com/user-attachments/assets/97d3820a-e3b3-4c75-8445-13e672fc0bff)

## Dataset
### Input File
- **`Dane_3_2_Mandrill.xlsx`**
Contains the textual data categorized into two sheets:
    1. **`GOTOWE DANE`**: Data labeled as "Mandrill".
    2. **`GOTOWE INNE`**: Data labeled as "inne".

The data from both sheets is combined and processed into a single dataframe with the following columns:
- **Post**: Contains the text/content of the post.
- **label**: Indicates the category, either `"Mandrill"` or `"inne"`.

## Project Pipeline
### 1. Data Loading
The dataset is loaded using `pandas` from an Excel file. The two sheets are merged, and labels are assigned to indicate their categories.
### 2. Data Preprocessing
The `Post` column undergoes TF-IDF vectorization using scikit-learn's `TfidfVectorizer`. Features are limited to the top 5000 terms, and English stop words are removed to improve model performance.
### 3. Machine Learning Models
#### a) Naive Bayes Classifier
- A simple baseline model trained to classify posts.
- Evaluated using a classification report and confusion matrix.

#### b) Logistic Regression
- A logistic regression model trained to improve classification performance.
- Evaluated using accuracy, classification report, confusion matrix, and ROC-AUC curve.

### 4. Evaluation Metrics
- **Classification Report**: Precision, Recall, F1-Score for both categories.
- **Accuracy**: General performance of the model.
- **Confusion Matrix**: Detailed breakdown of true positives, false positives, etc.
- **ROC Curve**: Visual illustration of the trade-off between TPR and FPR, along with AUC value for predictive power.

### 5. Predictions on New Data
Test cases are provided with unseen posts, classified into "Mandrill" or "inne" classes using both trained models.
## Project Highlights
### a) Key Features
- **Text Classification**: Enables categorizing posts into meaningful labels.
- **ROC-AUC Analysis**: Evaluates model's capability for distinguishing between categories.
- **Visualization**: Confusion matrix and ROC-AUC curve provide insights into model performance.

### b) Tools and Libraries
- **Libraries**:
    - NumPy, pandas for data manipulation.
    - scikit-learn for machine learning and evaluation tools.
    - Matplotlib for data visualization.

- **Key Techniques**:
    - TF-IDF vectorization
    - Naive Bayes and Logistic Regression classifiers
    - Label conversion for ROC analysis using LabelBinarizer

## How to Run
1. Install the required libraries using pip:
