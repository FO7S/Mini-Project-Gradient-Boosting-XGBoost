# 🎮 Gaming Addiction Risk Prediction
## Using Gradient Boosting, XGBoost

This project applies machine learning classification models to predict **gaming addiction risk level** based on behavioral, physical, social, and mental-health-related features.

The notebook covers the full workflow:
- Data understanding
- Missing value handling
- Data leakage removal
- Exploratory Data Analysis (EDA)
- Preprocessing with pipelines
- Baseline modeling
- Hyperparameter tuning
- Model comparison
- Feature importance analysis

---

## 📌 Project Objective

The goal of this project is to build a classification system that can predict the severity of gaming addiction risk using real-world user-related features such as:

- daily gaming hours
- sleep habits
- academic/work performance
- social isolation
- spending behavior
- physical symptoms
- years of gaming

The target variable is:

- `gaming_addiction_risk_level`

This is a **multiclass classification** problem.

---

## 📂 Dataset Overview

The dataset contains **1000 rows** and **27 columns**.

### Main feature groups:
- **Behavioral features**: gaming hours, spending, years gaming
- **Sleep-related features**: sleep hours, sleep quality, sleep disruption
- **Social features**: isolation score, face-to-face social hours
- **Physical indicators**: eye strain, back/neck pain, weight change
- **Academic / work indicators**: GPA, work productivity
- **Mood-related indicators**: mood state, mood swing frequency

### Missing values found:
- `grades_gpa` → 246 missing values
- `work_productivity_score` → 326 missing values

---

## 🧹 Data Preprocessing

The following preprocessing steps were applied:

### 1) Handling missing values
Median imputation was used for:
- `grades_gpa`
- `work_productivity_score`

### 2) Removing data leakage
The following columns were removed because they directly describe addiction behavior and may leak the target:

- `withdrawal_symptoms`
- `loss_of_other_interests`
- `continued_despite_problems`

### 3) Removing irrelevant columns
The notebook removes:
- `record_id`
- `primary_game`

### 4) Encoding target labels
The target classes were encoded as:

- `0 -> High`
- `1 -> Low`
- `2 -> Moderate`
- `3 -> Severe`

> Note: Label encoding order is alphabetical, not ordinal severity order.

### 5) Train-test split
- **80% training**
- **20% testing**
- Stratified split used to preserve class distribution

### 6) Feature preprocessing
A preprocessing pipeline was used:
- **Numerical features** → median imputation
- **Categorical features** → most frequent imputation + one-hot encoding

---

## 📊 Exploratory Data Analysis (EDA)

The notebook includes several visual analyses:

### Target distribution
The dataset is imbalanced, with **Low** risk being the dominant class.

### Daily gaming hours distribution
Most users spend around **4 to 8 hours** gaming daily, with a slight right skew.

### Gaming hours vs sleep hours
A negative relationship appears between gaming time and sleep duration.

### Social isolation vs addiction risk
Higher addiction risk levels tend to show higher social isolation scores.

### Correlation heatmap
Important relationships appear among:
- gaming hours
- sleep hours
- social isolation
- spending and behavior-related patterns

---

## 🤖 Models Used

The following classification models were implemented:

1. **Gradient Boosting Classifier**
2. **XGBoost Classifier**
3. **Tuned XGBoost**
4. **K-Nearest Neighbors (KNN)**

---

## 📈 Baseline Model Results

### Gradient Boosting
- **Accuracy:** 0.690
- **Precision:** 0.705877
- **Recall:** 0.690
- **F1 Score:** 0.688236

### XGBoost
- **Accuracy:** 0.675
- **Precision:** 0.674328
- **Recall:** 0.675
- **F1 Score:** 0.672377

---

## 🚀 Tuned XGBoost

XGBoost was further improved using:
- `RandomizedSearchCV`
- `StratifiedKFold`
- `weighted F1 score`
- `sample_weight` with balanced class weights

### Best parameters found
```python
{
    "model__subsample": 1.0,
    "model__n_estimators": 250,
    "model__min_child_weight": 1,
    "model__max_depth": 4,
    "model__learning_rate": 0.05,
    "model__colsample_bytree": 0.9
}
