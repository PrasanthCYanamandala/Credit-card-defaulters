# Credit Card Default Prediction using R  

## 📌 Project Description  
This project analyzes **credit card default risk** using **machine learning models in R**. The dataset consists of **30,000 credit card clients** from Taiwan, collected between **April 2005 and September 2005**, and includes demographic details, credit history, payment records, and bill statements.  

The primary objective is to **predict whether a customer will default on their credit card payment in the following month**, helping financial institutions mitigate risks and make informed decisions.  

---

## 📊 Dataset Information  
- **Source**: [Kaggle - Default of Credit Card Clients Dataset](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset)  
- **Target Variable**: `DEFAULT` (1 = Defaulter, 0 = Non-Defaulter)  
- **Independent Variables**:  
  - **Demographics**: Age, Gender, Education, Marital Status  
  - **Credit Information**: Bill Amount, Payment Amount, Credit Limit  
  - **Payment History**: Past defaults and late payments  

---

## 🎯 Business Objective  
- **Identify high-risk customers** likely to default.  
- **Minimize financial loss** by reducing false negatives (defaulters misclassified as non-defaulters).  
- **Optimize credit approval policies** for better decision-making.  

---

## 🛠️ Project Workflow  
1️⃣ **Data Preprocessing**:  
   - Handle missing values  
   - Remove highly correlated features  
   - Scale numerical variables  
   - Encode categorical variables  

2️⃣ **Exploratory Data Analysis (EDA)**:  
   - Analyze trends in defaulters vs. non-defaulters  
   - Correlation analysis  
   - Data visualization  

3️⃣ **Model Training & Evaluation**:  
   - Logistic Regression  
   - Random Forest  
   - Boosting Model  
   - Neural Networks  

4️⃣ **Performance Comparison**:  
   - Accuracy  
   - ROC-AUC Score  
   - False Positive Rate  

---

## 🏆 Key Findings  
- **Logistic Regression** performed best with a **ROC-AUC score of 72.67%**.  
- **False Positive Rate** is critical in financial decision-making.  
- **Neural Networks** showed the highest accuracy but had a higher False Positive Rate.  
- **Future improvements** could include hyperparameter tuning and additional feature engineering.  

---

## 👨‍💻 Team Members  
- **Martin Aguirre**  
- **Vishruth Acharya**  
- **Yuvaraja Reddy Ambati**  
- **Varun Namala**  
- **Prasanth Chowdary Yanamandala**  

---

## 📜 License  
This project is for academic purposes only.  

