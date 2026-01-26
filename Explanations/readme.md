
---


## 🔹 Steps followed in Experiment 2

---

### 1️⃣ Dataset Loading

Datasets were loaded using Pandas:

```python
pd.read_csv()
```

This allows structured tabular data handling.

---

### 2️⃣ Dataset Inspection

Basic inspection includes:

* number of rows and columns
* column names
* data types
* target variable identification

This helps determine whether the task is:

* classification
* regression

---

### 3️⃣ Handling Missing Values

Missing values can affect model performance.

Methods used:

* **Numerical features:** median
* **Categorical features:** most frequent value

Median is preferred because it is robust to outliers.

---

### 4️⃣ Statistical Summary

Using:

```python
df.describe()
```

We analyze:

* mean
* standard deviation
* minimum
* maximum

This shows data spread and skewness.

---

### 5️⃣ Data Visualization

Visualization gives intuitive understanding.

#### Histograms

* show feature distribution
* detect skewed data

#### Boxplots

* identify outliers
* based on Interquartile Range (IQR)

[
IQR = Q3 - Q1
]

Outliers lie outside:

[
Q1 - 1.5 \times IQR
]

---

#### Correlation Heatmap

Correlation coefficient:

[
r \in [-1, +1]
]

* +1 → strong positive relation
* −1 → strong negative relation
* 0 → no relation

Helps remove redundant features.

---

### 6️⃣ Feature Encoding

Machine learning models work only with numbers.

Categorical variables were converted using:

* Label Encoding
* One-Hot Encoding

---

### 7️⃣ Feature Scaling

StandardScaler was used:

[
z = \frac{x - \mu}{\sigma}
]

Scaling is important for:

* KNN
* SVM
* distance-based algorithms

---

### 8️⃣ Model Implementation

Based on problem type:

| Dataset  | Algorithm     |
| -------- | ------------- |
| Iris     | KNN           |
| Diabetes | SVM           |
| Spam     | Naive Bayes   |
| Loan     | Random Forest |

---

### 9️⃣ Model Evaluation

Metrics used:

* Accuracy
* Precision
* Recall
* F1-score
* R² score (for regression)

---

## 🔹 Outcome of Experiment 2

* Understood dataset behavior
* Learned preprocessing techniques
* Identified suitable algorithms
* Built strong foundation for further experiments

---

# ✅ **EXPERIMENT 3 – DETAILED EXPLANATION**

---

# 🔹 Experiment 3

## *Regression Analysis using Linear and Regularized Models*

---

## 🔸 Objective

To predict a **continuous target variable** using:

* Linear Regression
* Ridge Regression
* Lasso Regression
* Elastic Net Regression

and compare their performance.

---

## 🔹 Why regression?

Regression is used when:

* output is numeric
* values are continuous

Example:

* loan amount
* salary
* house price

---

## 🔸 Linear Regression

Linear regression assumes:

[
y = w^Tx + b
]

It minimizes:

[
\sum (y - \hat{y})^2
]

---

### Problem with Linear Regression

* sensitive to noise
* large coefficients
* easily overfits
* unstable with correlated features

---

## 🔹 Regularization

Regularization solves overfitting by adding a penalty term.

[
Loss = Error + \lambda \times Penalty
]

---

### Ridge Regression (L2)

[
\lambda \sum w_i^2
]

* shrinks coefficients
* keeps all features
* improves stability

---

### Lasso Regression (L1)

[
\lambda \sum |w_i|
]

* sets weak coefficients to zero
* performs feature selection

---

### Elastic Net

Combination of both:

[
\lambda_1 \sum |w_i| + \lambda_2 \sum w_i^2
]

---

## 🔸 Hyperparameter Tuning

GridSearchCV was used to find best:

* alpha
* l1_ratio

using **5-fold cross-validation**.

---

## 🔹 Model Evaluation Metrics

* MAE
* MSE
* RMSE
* R² Score

---

## 🔸 Visualization

* target distribution
* correlation heatmap
* coefficient comparison
* training vs validation error
* residual analysis

---

## 🔹 Overfitting & Underfitting

* Similar train/test scores → no overfitting
* Moderate R² (~0.55) → slight underfitting

Linear models cannot capture complex relationships.

---

## 🔹 Bias–Variance Tradeoff

| Model             | Observation      |
| ----------------- | ---------------- |
| Linear Regression | High bias        |
| Ridge             | Reduced variance |
| Lasso             | Feature sparsity |
| Elastic Net       | Balanced         |

---

## 🔹 Final Conclusion

* Regularization improves stability
* Lasso enables feature selection
* Elastic Net provides best balance
* Linear models have limited capacity

To improve performance:

* Random Forest
* Gradient Boosting
* XGBoost

can be explored.

---
