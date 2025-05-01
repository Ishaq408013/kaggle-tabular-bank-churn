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

## Summary of Workdone

### Data

* Data:
  * Type:
    * Input: CSV files containing customer demographic and financial activity features
    * Output: A binary label (`Exited`) representing churn (1) or retention (0)
  * Size:
    * Training set: ~8,000 entries
    * Test set: ~4,000 entries
  * Instances:
    * 64% used for training, 16% for validation, 20% for testing

#### Preprocessing / Clean up

* Dropped identifier columns (`id`, `CustomerId`, `Surname`)
* One-hot encoded categorical features (`Gender`, `Geography`)
* Scaled numerical columns using `StandardScaler`
* Verified there were no missing values
* Detected and described outliers using the 1.5×IQR method
* Verified mild class imbalance (about 20% churned)

#### Data Visualization

* Plotted histograms of numerical features grouped by class (`Exited`)
* Plotted stacked bar graphs for categorical features
* Observed `Age`, `Balance`, and `IsActiveMember` as strong indicators of churn

### Problem Formulation

* **Input**: Cleaned tabular data (numerical + one-hot encoded features)
* **Output**: Binary target variable (`Exited`)
* **Models**:
  * Random Forest Classifier
  * Chosen for its high performance on tabular datasets and minimal tuning
* **Hyperparameters**:
  * Default parameters (`random_state=42`)

### Training

* **Software**: Jupyter Notebook, Python 3.12, scikit-learn
* **Hardware**: MacBook Pro (no GPU)
* **Duration**: Less than 1 minute
* **Stopping Criteria**: Model was stable with default settings
* **Challenges**: None significant; model trained efficiently without overfitting

### Performance Comparison

* **Metrics Used**: Accuracy, F1-Score, Confusion Matrix
* **Validation Results**:
  * Accuracy: 84%
  * F1 Score: 0.72
* **Submission File**: `submission/submission.csv` generated from test data

### Conclusions

* Random Forest achieved reliable performance with minimal tuning
* Strongest predictive features were `Age`, `Balance`, and `IsActiveMember`
* Feature engineering and model tuning can further improve performance

### Future Work

* Add models such as XGBoost or LightGBM for comparison
* Use SHAP values or permutation importance for explainability
* Apply SMOTE or class weighting to address imbalance
* Perform hyperparameter tuning via GridSearchCV



### Overview of files in repository

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

