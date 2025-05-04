![](UTA-DataScience-Logo.png)
# Bank Customer Churn Prediction

* **One Sentence Summary**  
  This repository holds an attempt to predict customer churn using tabular data from the [Playground Series - Season 4, Episode 1](https://www.kaggle.com/competitions/playground-series-s4e1) Kaggle challenge.

## Overview

* **Definition of the tasks / challenge**  
  The task is to predict whether a bank customer will close their account (Exited = 1) based on demographic and account-related features such as age, balance, credit score, and account activity.

* **Approach**  
  This project formulates the problem as a binary classification task. Data preprocessing included one-hot encoding categorical features, scaling numerical features, and removing irrelevant ID columns. A Random Forest Classifier was trained and evaluated on split validation data.

* **Summary of the performance achieved**  
  The model achieved a validation accuracy of approximately 84% and an F1-score of 0.72. 

---

## Summary of Workdone

### Data

* **Type**:
  * Input: Tabular CSV files with demographic and financial information
  * Output: Binary target column `Exited`
* **Size**:
  * ~8,000 rows (train), ~4,000 rows (test)
* **Splits**:
  * 64% train, 16% validation, 20% test

---

### Preprocessing and Clean-up

- Dropped ID-related columns: `id`, `CustomerId`, `Surname`
- One-hot encoded: `Gender`, `Geography`
- Scaled numerical columns using `StandardScaler`
- Detected outliers using IQR method (not removed)
- Verified no missing values
- Confirmed mild class imbalance (~20% churned)
  
  The figure below shows the distribution of four key numerical features—Credit Score, Age, Balance, and Estimated Salary—before and after applying `StandardScaler`. While the shape of each distribution remains unchanged, the scaling standardizes the feature ranges around a mean of 0 and standard deviation of 1. This is especially important for models sensitive to feature magnitudes, like logistic regression or SVMs, and ensures that features contribute equally.
<img width="801" alt="Screenshot 2025-05-04 at 9 21 19 AM" src="https://github.com/user-attachments/assets/45dec799-f6fe-440b-b081-1381be80d515" />




---

### Feature Summary Table

<img width="815" alt="Screenshot 2025-05-02 at 11 33 14 AM" src="https://github.com/user-attachments/assets/5b78cf9b-36eb-421d-8412-86e94e98461b" />

> This table provides an overview of each feature's type, range, and missing values. It helped guide data cleaning, scaling, and encoding decisions.

---

### Data Visualization

#### Numerical Feature Distributions by Class
This visualization compares the distributions of key numerical features (e.g., `Age`, `CreditScore`, `Balance`, `EstimatedSalary`) between customers who stayed (`Exited = 0`) and those who churned (`Exited = 1`). Notably, churned customers tended to be older, less likely to be active members, and had slightly higher balances.

<img width="1022" alt="Screenshot 2025-05-02 at 11 30 37 AM" src="https://github.com/user-attachments/assets/4dff2ea7-2681-494f-89b3-1c5669899342" />


Visualized feature distributions across churn vs retained customers:


#### Categorical Feature Distributions

Comparison of categorical variables between churn classes:

The chart below illustrates the distribution of churn (`Exited`) across categorical features such as `Surname`, `Geography`, and `Gender`. As expected, `Surname` shows no predictive power, while `Geography` and `Gender` demonstrate clearer trends—particularly with higher churn rates in Germany and among female customers. These insights guided feature selection and one-hot encoding during preprocessing.

<img width="926" alt="Screenshot 2025-05-02 at 11 34 56 AM" src="https://github.com/user-attachments/assets/9913afa4-306d-4cf1-a533-43c4cb5c72cb" />


---

### Problem Formulation

* **Input**: Cleaned feature matrix (numerical + categorical)
* **Output**: Binary label (`Exited`)
* **Model**: Random Forest Classifier (default hyperparameters)
* **Why**: Works well on structured/tabular data, minimal tuning required

---

### Training

* **Tools**: Python 3.12, scikit-learn, Jupyter Notebook
* **Platform**: MacBook Pro (no GPU)
* **Training Time**: < 1 minute
* **Split**: Train / Validation / Test

---

### Performance Evaluation

- **Accuracy**: 85.7%
- **F1 Score**: 0.72 (overall)
- **Evaluation Tools**: Accuracy, F1, confusion matrix

#### Classification Report & Confusion Matrix
The classification report and confusion matrix below illustrate the model’s performance on the validation set. The model achieved an overall accuracy of **85.7%**, with an F1-score of **0.91** for the majority class (retained customers) and **0.61** for the minority class (churned customers). The confusion matrix shows the model correctly predicted most retained customers, though it missed some churned cases—highlighting the impact of class imbalance.

<img width="546" alt="Screenshot_Confusion_Matrix" src="https://github.com/user-attachments/assets/ccff28f4-736c-4787-a7e1-0de8fc21fbd2" />


---

### Conclusions

* Random Forest gave strong results with default tuning
* Most predictive features:
  * Age
  * Balance
  * IsActiveMember
* Future improvements could come from ensemble methods or tuning

---

### Future Work

- Try XGBoost, LightGBM
- Use SHAP values for explainability
- Address class imbalance with SMOTE
- Tune hyperparameters with GridSearchCV

---

## Repository Overview



* `data/train.csv`: Training dataset  
* `data/test.csv`: Test dataset  
* `notebooks/Kaggle_Tabular_Data.ipynb`: Full analysis, training, and submission pipeline  
* `submission/submission.csv`: Final predictions  
* `images/`: Folder for any visualization outputs (optional)  
* `.gitignore`: Hides checkpoint and system files  
* `README.md`: This project summary

### Software Setup

* Python 3.12
* Packages required:
* pandas
* scikit-learn
* matplotlib
* seaborn


### Data

* Source: [Kaggle Playground Series - Season 4, Episode 1](https://www.kaggle.com/competitions/playground-series-s4e1)
* Files:
- `train.csv`
- `test.csv`
- `sample_submission.csv`

### Training

* Execute all cells in `Kaggle_Tabular_Data.ipynb` to reproduce preprocessing, training, and prediction steps

#### Performance Evaluation

* Validation performance is shown via:
- Accuracy
- F1-score
- Confusion Matrix
* Final predictions saved to `submission/submission.csv`

## Citations

* Kaggle Challenge: https://www.kaggle.com/competitions/playground-series-s4e1  
* scikit-learn Documentation: https://scikit-learn.org  

