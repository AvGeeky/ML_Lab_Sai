
# Machine Learning Algorithms Laboratory  
## Experiment 2 – Exploratory Data Analysis and Basic Machine Learning Models

---

## Student Information

- **Name:** Saipranav M  
- **Register Number:** 3122235001110  
- **Course:** B.E. Computer Science and Engineering  
- **Semester:** VI  
- **Academic Year:** 2025–2026 (Even)  
- **Subject Code:** UCS2612  

---

## 1. Aim

To perform exploratory data analysis (EDA) on real-world datasets, understand data characteristics,
apply preprocessing techniques, visualize feature relationships, and implement suitable machine
learning algorithms based on the problem type.

---

## 2. Objective

- Understand dataset structure and feature types  
- Identify missing values and outliers  
- Visualize feature distributions and correlations  
- Apply preprocessing techniques  
- Implement suitable machine learning models  

---

## 3. Datasets Used

The following datasets were used for this experiment:

1. **Iris Dataset** – Multi-class classification  
2. **Diabetes Dataset** – Binary classification  
3. **Loan Dataset** – Regression  
4. **Email Spam Dataset** – Text classification  

Each dataset represents a different machine learning problem.

---

## 4. Software and Libraries Used

- **Python 3**
- **NumPy** – numerical computation  
- **Pandas** – data handling  
- **Matplotlib** – plotting graphs  
- **Seaborn** – statistical visualization  
- **Scikit-learn** – machine learning algorithms  

## 5. Exploratory Data Analysis (EDA)

Exploratory Data Analysis is performed to understand the dataset before building machine learning models.

EDA helps in identifying:

- data distribution  
- missing values  
- outliers  
- feature relationships  

---

### 5.1 Dataset Inspection

The following properties were examined:

- Number of rows and columns  
- Column names  
- Data types of attributes  
- Identification of target variable  

---

### 5.2 Missing Value Analysis

Missing values were detected using:

```python
df.isnull().sum()
````

Handling strategy:

* **Numerical attributes:** Median imputation
* **Categorical attributes:** Mode imputation

---

### 5.3 Statistical Summary

The statistical summary includes:

* Mean
* Median
* Standard deviation
* Minimum and maximum values

This analysis helps in understanding feature spread and skewness.

---

## 6. Data Visualization

Visualization provides intuitive insights into the dataset.

---

### 6.1 Histogram

* Displays frequency distribution of features
* Helps identify skewed data

---

### 6.2 Boxplot

Boxplots were used to detect outliers.

The Interquartile Range (IQR) is defined as:

[
IQR = Q3 - Q1
]

Outliers are values lying outside:

[
Q1 - 1.5 \times IQR \quad \text{or} \quad Q3 + 1.5 \times IQR
]

---

### 6.3 Correlation Heatmap

Correlation between numerical variables was measured using the Pearson correlation coefficient:

[
r = \frac{Cov(X, Y)}{\sigma_X \sigma_Y}
]

* Values close to **+1** or **−1** indicate strong correlation
* Helps identify redundant features

---

### 6.4 Scatter Plot

Scatter plots were used to visualize:

* relationship between features and target variable
* separation between classes

---

## 7. Feature Engineering

---

### 7.1 Encoding Categorical Variables

Categorical attributes were converted into numerical format using:

* Label Encoding
* One-Hot Encoding

This enables machine learning algorithms to process categorical data.

---

### 7.2 Feature Scaling

Numerical features were standardized using:

[
z = \frac{x - \mu}{\sigma}
]

Feature scaling is important for:

* distance-based algorithms
* gradient-based optimization

---

## 8. Machine Learning Models Implemented

Different algorithms were selected based on the problem type.

---

### 8.1 Classification Models

* Logistic Regression
* K-Nearest Neighbors (KNN)
* Support Vector Machine (SVM)
* Naive Bayes

---

### 8.2 Regression Models

* Linear Regression
* Random Forest Regressor

---

## 9. Model Selection Strategy

| Problem Type               | Algorithm Used                    |
| -------------------------- | --------------------------------- |
| Binary classification      | Logistic Regression / SVM         |
| Multi-class classification | KNN                               |
| Text classification        | Naive Bayes                       |
| Regression                 | Linear Regression / Random Forest |

---

## 10. Performance Metrics

---

### 10.1 Classification Metrics

* Accuracy
* Precision
* Recall
* F1-score

[
Accuracy = \frac{TP + TN}{TP + TN + FP + FN}
]

[
F1 = \frac{2 \cdot Precision \cdot Recall}{Precision + Recall}
]

---

### 10.2 Regression Metrics

* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)
* R² Score

[
R^2 = 1 - \frac{SS_{res}}{SS_{tot}}
]

---

## 11. Observations

* Data visualization helped detect outliers and skewed features
* Correlation analysis reduced redundant features
* Feature scaling improved algorithm performance
* Different datasets required different algorithms

---

## 12. Results Summary

| Dataset  | Task Type                  | Algorithm     |
| -------- | -------------------------- | ------------- |
| Iris     | Multi-class classification | KNN           |
| Diabetes | Binary classification      | SVM           |
| Spam     | Text classification        | Naive Bayes   |
| Loan     | Regression                 | Random Forest |

---

## 13. Learning Outcomes

* Understood the importance of EDA
* Learned preprocessing and visualization techniques
* Implemented multiple ML algorithms
* Gained experience in model evaluation

---

## 14. Conclusion

Exploratory Data Analysis plays a crucial role in machine learning model development.
Proper data understanding, visualization, and preprocessing significantly improve model performance.

Different datasets require different algorithms, and model selection must be based on
the nature of the problem rather than accuracy alone.

---

## 15. References

* [https://scikit-learn.org](https://scikit-learn.org)
* [https://pandas.pydata.org](https://pandas.pydata.org)
* [https://matplotlib.org](https://matplotlib.org)
* [https://www.kaggle.com](https://www.kaggle.com)

---

