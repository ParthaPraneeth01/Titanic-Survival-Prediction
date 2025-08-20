# 🛳️ Titanic Survival Prediction (Kaggle Competition)

This repository contains my complete solution to the **[Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic)** competition on Kaggle.  
The Titanic dataset is a beginner-friendly yet challenging dataset widely used in machine learning. The goal is to build a model that predicts which passengers survived the Titanic shipwreck.

---

## 📌 Problem Statement
On April 15, 1912, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew.  
One of the reasons this tragedy resulted in such loss of life was that there were not enough lifeboats.  
While luck played a role, some groups of people had a higher chance of survival than others, such as women, children, and those in higher classes.  

**Task:** Use machine learning to predict survival based on passenger data (such as age, sex, ticket class, family size, etc.).

---

## 📊 Dataset Information
The dataset consists of two main files provided by Kaggle:
- **train.csv** — contains labeled data (including the `Survived` column).  
- **test.csv** — contains unlabeled data where we need to predict survival.  

**Key Features:**
- `Pclass` – Ticket class (1st, 2nd, 3rd)  
- `Sex` – Gender (male/female)  
- `Age` – Age of passenger  
- `SibSp` – Number of siblings/spouses aboard  
- `Parch` – Number of parents/children aboard  
- `Fare` – Passenger fare  
- `Embarked` – Port of embarkation (C, Q, S)  
- `Survived` – Survival status (0 = No, 1 = Yes) *(only in train.csv)*  

---

## 🛠️ Approach & Methodology
1. **Data Cleaning & Preprocessing**
   - Handled missing values (Age, Embarked, Fare).  
   - Converted categorical features (`Sex`, `Embarked`) into numeric values.  
   - Normalized/standardized continuous features.  

2. **Feature Selection**
   - Considered key predictors: `Pclass`, `Sex`, `Age`, `Fare`, `SibSp`, `Parch`, `Embarked`.  
   - Created new features like `FamilySize = SibSp + Parch + 1`.  

3. **Model Training**
   - Used **k-Nearest Neighbors (k-NN)** classifier.  
   - Optimized hyperparameters (`n_neighbors`, `weights`, `p`) using **GridSearchCV**.  

4. **Model Evaluation**
   - Evaluated using cross-validation on training data.  
   - Selected best k-NN model for predictions.  

5. **Prediction & Submission**
   - Generated predictions on test data.  
   - Created `titanic_knn_submission.csv` for Kaggle submission.  

---

## 📂 Repository Structure
Titanic-Survival-Prediction<br>
│── Titanic_Survival_Prediction_using_k_NN_completed.ipynb # Jupyter Notebook<br>
│── titanic_knn_submission.csv # Kaggle submission file<br>
│── README.md # Project documentation<br>
│── requirements.txt # (optional) dependencies

---
## 🚀 How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/Titanic-Survival-Prediction.git
   cd Titanic-Survival-Prediction
2.Install dependencies:
    pip install -r requirements.txt <br>
3.Place the Kaggle dataset (train.csv, test.csv) inside the project folder.<br>
4.Run the Jupyter notebook:
    jupyter notebook Titanic_Survival_Prediction_using_k_NN_completed.ipynb
    
---
## 📊 Results
- **Best Model:** k-NN Classifier (tuned with GridSearchCV).  
- **Kaggle Public Leaderboard Score:** `0.75119`.  
- Generated a valid Kaggle submission file: `titanic_knn_submission.csv`.  

---

## 📈 Possible Improvements
To push accuracy above **0.80+**, the following improvements can be made:

### 🔹 Advanced Feature Engineering
- Extract titles from names (`Mr`, `Mrs`, `Miss`, `Master`).  
- Create an `IsAlone` feature (whether the passenger was traveling alone).  
- Bucket `Age` and `Fare` into categorical ranges.  

### 🔹 Use Stronger Models
- Logistic Regression, Random Forest, XGBoost, LightGBM.  
- Try ensemble methods (voting, stacking multiple models).  

---

## 🛠️ Tech Stack
- Python  
- Jupyter Notebook  
- **pandas** – Data manipulation  
- **numpy** – Numerical computing  
- **matplotlib / seaborn** – Visualization  
- **scikit-learn** – Machine Learning (KNN, GridSearchCV)  

---

## ✨ Acknowledgements
- Kaggle Titanic Competition  
- Libraries: scikit-learn, pandas, numpy, matplotlib  

---

## 📧 Contact
**Author:** Partha Praneeth Reddy Pocham Reddy <br> 
**Gmail:** parthapraneeth01@gmail.com <br>
**Kaggle:** https://www.kaggle.com/parthapraneeth <br>
**LinkedIn** www.linkedin.com/in/partha-praneeth

