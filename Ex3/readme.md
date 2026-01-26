# Machine Learning Algorithms Laboratory  
## Experiment 3 – Regression Analysis using Linear and Regularized Models

---

## Student Information

- **Name:** Saipranav M  
- **Register Number:** 3122235001110  
- **Course:** B.E. Computer Science and Engineering  
- **Semester:** VI  
- **Academic Year:** 2025–2026 (Even)  
- **Subject Code:** UCS2612  

---

# 1. Aim

To implement linear and regularized regression models for predicting a continuous target variable,
evaluate their performance using appropriate regression metrics, visualize model behavior,
and analyze overfitting, underfitting, and bias–variance characteristics.

---

# 2. Dataset Description

A real-world loan prediction dataset obtained from Kaggle was used for this experiment.

The dataset contains both numerical and categorical attributes related to loan applications,
such as applicant income, age, expenses, dependents, and requested loan amount.

### Target Variable

- **Loan Amount Sanctioned (USD)**

---

# 3. Tools and Libraries Used

- **Python**
- **NumPy** – numerical computation  
- **Pandas** – data manipulation  
- **Matplotlib & Seaborn** – visualization  
- **Scikit-learn** – machine learning models  

---

# 4. Preprocessing Steps

### 4.1 Handling Missing Values

- Numerical features were imputed using the **median** strategy.
- Categorical features were imputed using the **most frequent value**.

---

### 4.2 Encoding Categorical Variables

Categorical attributes were converted into numerical form using **One-Hot Encoding**.

This avoids ordinal bias and allows models to interpret categories independently.

---

### 4.3 Feature Scaling

Numerical features were standardized using **StandardScaler**:

\[
z = \frac{x - \mu}{\sigma}
\]

Scaling is essential for regularized models where coefficient magnitude affects learning.

---

# 5. Models Implemented

The following regression models were implemented and compared:

1. **Linear Regression**
2. **Ridge Regression (L2 Regularization)**
3. **Lasso Regression (L1 Regularization)**
4. **Elastic Net Regression (L1 + L2 Regularization)**

---

# 6. Mathematical Background

---

## 6.1 Linear Regression

\[
y = w^Tx + b
\]

The model attempts to minimize:

\[
\sum (y - \hat{y})^2
\]

---

## 6.2 Regularization

Regularization introduces a penalty term to reduce overfitting.

\[
Loss = Error + \lambda \times Penalty
\]

---

### L2 Regularization (Ridge)

\[
\lambda \sum w_i^2
\]

- Shrinks coefficients
- Improves stability
- Retains all features

---

### L1 Regularization (Lasso)

\[
\lambda \sum |w_i|
\]

- Performs feature selection
- Sets weak coefficients to zero

---

### Elastic Net

\[
\lambda_1 \sum |w_i| + \lambda_2 \sum w_i^2
\]

Balances sparsity and stability.

---

### Relationship between α and C

\[
C = \frac{1}{\alpha}
\]

| Value | Effect |
|------|------|
| Small C | Strong regularization |
| Large C | Weak regularization |

---

# 7. Hyperparameter Tuning

Hyperparameter optimization was performed using **GridSearchCV** with **5-fold cross-validation**.

### Parameters tuned:

- Ridge: `alpha`
- Lasso: `alpha`
- Elastic Net: `alpha`, `l1_ratio`

The model achieving the highest average cross-validation score was selected.

---

# 8. Performance Metrics

The following evaluation metrics were used:

- **Mean Absolute Error (MAE)**
- **Mean Squared Error (MSE)**
- **Root Mean Squared Error (RMSE)**
- **Coefficient of Determination (R²)**

---

# 9. Visualizations Performed

- Target variable distribution  
- Feature vs target scatter plots  
- Correlation heatmap  
- Coefficient comparison bar chart  
- Training vs validation error curve  
- Predicted vs actual values  
- Residual analysis plot  

These visualizations provide insight into data distribution,
model behavior, and prediction errors.

---

# 10. Results Summary

| Model | Test R² |
|------|------|
| Linear Regression | 0.516 |
| Ridge Regression | 0.519 |
| Lasso Regression | 0.519 |
| Elastic Net Regression | 0.542 |

Elastic Net produced the highest R² score, although all models performed similarly.

---

# 11. Overfitting and Underfitting Analysis

- Training and testing performances were close.
- No severe overfitting was observed.
- Moderate R² values (~0.52–0.54) indicate **mild underfitting**.

This suggests that linear models cannot fully capture the complex relationships present in the dataset.

---

# 12. Bias–Variance Analysis

| Model | Observation |
|------|------|
| Linear Regression | High bias |
| Ridge Regression | Reduced variance |
| Lasso Regression | Feature sparsity |
| Elastic Net | Balanced bias–variance |

Regularization significantly improved coefficient stability.

---

# 13. Observations

- Regularization reduced coefficient explosion.
- Ridge and Elastic Net produced more stable models.
- Lasso successfully eliminated weak features.
- Model performance was limited by linear assumptions.

---

# 14. Conclusion

Linear and regularized regression models were successfully implemented and evaluated.

Although regularization improved stability and reduced variance,
overall prediction accuracy remained limited due to the linear nature of the models.

For improved performance, future work may include:

- Random Forest Regression  
- Gradient Boosting  
- XGBoost  
- Neural Networks  

---

# 15. Dataset Reference

- Kaggle – Predict Loan Amount Dataset  
  https://www.kaggle.com/datasets/phileinsophos/predict-loan-amount-data

---

# 16. References

- https://scikit-learn.org/stable/modules/linear_model.html  
- https://scikit-learn.org/stable/modules/grid_search.html  

---

# ✅ End of Assignment
