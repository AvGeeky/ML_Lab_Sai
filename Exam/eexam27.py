
"""
sklearn
 ├── model_selection   → train_test_split, GridSearchCV
 ├── preprocessing     → StandardScaler, MinMaxScaler, OneHotEncoder
 ├── impute            → SimpleImputer
 ├── pipeline          → Pipeline
 ├── compose           → ColumnTransformer
 ├── metrics           → accuracy_score, f1_score, roc_curve
 ├── linear_model      → LogisticRegression, Ridge, Lasso
 ├── svm               → SVC
 ├── neighbors         → KNeighborsClassifier
 ├── naive_bayes       → GaussianNB, MultinomialNB
 ├── tree              → DecisionTreeClassifier
 └── ensemble          → RandomForestClassifier
 import sklearn
dir(sklearn)

 import sklearn.compose
dir(sklearn.compose)

model_selection
preprocessing
impute
pipeline
compose
metrics
linear_model
svm
neighbors
naive_bayes
tree
ensemble
 """


import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
# SKLEARN CORE
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import *
from sklearn.compose import *
from sklearn.impute import *
from sklearn.preprocessing import *

# MODELS
from sklearn.linear_model import *
from sklearn.svm import *
from sklearn.neighbors import *
from sklearn.naive_bayes import *
from sklearn.tree import *
from sklearn.ensemble import *
from sklearn.neural_network import *

# METRICS
from sklearn.metrics import *
# =====================================================
# 1. LOAD DATASET (CHANGE FOR LAB)
# =====================================================

# # Example dataset
# from sklearn.datasets import make_classification
#
# X_num, y = make_classification(
#     n_samples=600,
#     n_features=10,
#     random_state=42
# )
#
# X = pd.DataFrame(X_num, columns=[f"num_{i}" for i in range(10)])
# X["cat_1"] = np.random.choice(["A", "B", "C"], size=len(X))
#

df = pd.read_csv('spam.csv')
X = df.drop("class", axis = 1)
y = df["class"]

# DESCRIBE AND FIXING DATA
# df.describe()
# df = df.replace("?", np.nan)
#
# df = df.dropna(subset=["Property Price"])
# df["Property Price"] = pd.to_numeric(df["Property Price"], errors="coerce")

# EDA

#sns.pairplot(df[num_features])
# plt.show()

#sns.boxplot(x=df['target'], y=df['Income (USD)'])
# plt.show()

# Histogram of a numeric feature for skewness or gaussian distribution
# sns.histplot(df['Income (USD)'], kde=True)
# plt.show()

# Correlation Matrix
# sns.heatmap(df[num_features].corr(), annot=True, cmap='coolwarm')
# plt.show()

# Class Distribution (Number of samples per class)
# sns.countplot(x=df['target'])
# plt.show()


# =====================================================
# 2. CHANGE ONLY THIS LINE FOR DIFFERENT EXPERIMENTS
# =====================================================

EXPERIMENT = 4

"""
2 → Naive Bayes + KNN
3 → Regression
4 → Logistic + SVM
5 → Perceptron vs MLP
6 → Decision Tree + Random Forest
"""

# =====================================================
# 3. TRAIN TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =====================================================
# 4. PREPROCESSING
# =====================================================

numeric_features = X.select_dtypes(include=np.number).columns
categorical_features = X.select_dtypes(exclude=np.number).columns

# NOTE: If running MultinomialNB in Exp 2, change StandardScaler() to MinMaxScaler() below!
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])

# =====================================================
# 5. EXPERIMENT CONFIGURATION
# =====================================================

if EXPERIMENT == 2:
    task = "classification"

    # --- UNCOMMENT THE MODEL YOU WANT TO RUN ---

    model = KNeighborsClassifier()
    param_grid = {
        "model__n_neighbors": [3, 5, 7, 9],
        "model__algorithm": ["kd_tree", "ball_tree"]
    }

    # model = GaussianNB()
    # param_grid = {}

    # model = MultinomialNB()
    # param_grid = {} # Remember to change StandardScaler to MinMaxScaler above!

    # model = BernoulliNB()
    # param_grid = {}

elif EXPERIMENT == 3:
    task = "regression"

    # --- UNCOMMENT THE MODEL YOU WANT TO RUN ---

    model = ElasticNet(random_state=42)
    param_grid = {
        "model__alpha": [0.01, 0.1, 1, 10],
        "model__l1_ratio": [0.2, 0.5, 0.8]
    }

    # model = Ridge(random_state=42)
    # param_grid = {"model__alpha": [0.01, 0.1, 1, 10, 100]}

    # model = Lasso(random_state=42)
    # param_grid = {"model__alpha": [0.001, 0.01, 0.1, 1, 10]}

    # model = LinearRegression()
    # param_grid = {}

elif EXPERIMENT == 4:
    task = "classification"

    # --- UNCOMMENT THE MODEL YOU WANT TO RUN ---

    model = SVC(probability=True, random_state=42)
    param_grid = {
        "model__kernel": ["linear", "poly", "rbf", "sigmoid"],
        "model__C": [0.1, 1, 10, 100],
        "model__gamma": ["scale", "auto"],
        "model__degree": [2, 3, 4]
    }

    # model = LogisticRegression(max_iter=5000, random_state=42) # Increased max_iter
    # param_grid = {
    #     "model__penalty": ["l1", "l2"],
    #     "model__C": [0.01, 0.1, 1, 10, 100],
    #     "model__solver": ["liblinear", "saga"]
    # }

elif EXPERIMENT == 5:
    task = "classification"

    # --- UNCOMMENT THE MODEL YOU WANT TO RUN ---

    model = MLPClassifier(max_iter=1000, random_state=42)
    param_grid = {
        "model__hidden_layer_sizes": [(128,), (256, 128), (512, 256, 128)],
        "model__activation": ["relu", "tanh"],
        "model__solver": ["adam", "sgd"],
        "model__learning_rate_init": [0.01, 0.001],
        "model__batch_size": [32, 64]
    }

    # model = Perceptron(max_iter=1000, random_state=42)
    # param_grid = {}

elif EXPERIMENT == 6:
    task = "classification"

    # --- UNCOMMENT THE MODEL YOU WANT TO RUN ---

    model = RandomForestClassifier(random_state=42)  # Added random_state
    param_grid = {
        "model__n_estimators": [50, 100, 200],
        "model__max_depth": [None, 10, 20],
        "model__max_features": ["sqrt", "log2"],
        "model__bootstrap": [True, False]
    }

    # model = DecisionTreeClassifier(random_state=42)
    # param_grid = {
    #     "model__criterion": ["gini", "entropy"],
    #     "model__max_depth": [None, 5, 10, 20],
    #     "model__min_samples_split": [2, 5, 10],
    #     "model__min_samples_leaf": [1, 2, 4]
    # }

# =====================================================
# 6. PIPELINE
# =====================================================

pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

score = "accuracy" if task == "classification" else "r2"

grid = GridSearchCV(
    pipe,
    param_grid,
    cv=5,
    scoring=score,
    n_jobs=-1
)

print(f"\nTraining Model: {model.__class__.__name__}...\n")

start = time.time()
grid.fit(X_train, y_train)
train_time = time.time() - start

# =====================================================
# 7. PREDICTIONS
# =====================================================

start = time.time()
y_pred = grid.predict(X_test)
pred_time = time.time() - start

print("Best Parameters:", grid.best_params_)
print(f"Training Time: {train_time:.4f} seconds")
print(f"Prediction Time: {pred_time:.4f} seconds")

# =====================================================
# 8. CLASSIFICATION METRICS & PLOTS
# =====================================================

if task == "classification":

    # print("\n=== Classification Metrics ===")
    #
    # acc = accuracy_score(y_test, y_pred)
    # prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    # rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    # f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    #
    #
    # print(f"Accuracy:  {acc:.4f}")
    # print(f"Precision: {prec:.4f}")
    # print(f"Recall:    {rec:.4f}")
    # print(f"F1 Score:  {f1:.4f}")

    print(classification_report(y_test, y_pred))

    # 8A. Confusion Matrix
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap='Blues')
    plt.title(f"Confusion Matrix: {model.__class__.__name__}")
    plt.show()

    # 8B. ROC Curve & AUC
    if hasattr(grid.best_estimator_[-1], "predict_proba"):
        try:
            disp = RocCurveDisplay.from_estimator(grid, X_test, y_test)
            print(f"AUC: {disp.roc_auc:.4f}")
            plt.show()
        except ValueError:
            print("\n[Note: ROC Curve skipped. Target variable is likely multi-class instead of binary.]")

    # 8C. KNN Specific: Accuracy vs K Plot
    if isinstance(model, KNeighborsClassifier):
        k_vals = range(1, 15)
        scores = []

        for k in k_vals:
            knn = Pipeline([
                ("preprocessor", preprocessor),
                ("model", KNeighborsClassifier(n_neighbors=k))
            ])
            knn.fit(X_train, y_train)
            scores.append(knn.score(X_test, y_test))

        plt.plot(k_vals, scores, marker='o', linestyle='dashed', color='b')
        plt.show()

# =====================================================
# 9. REGRESSION METRICS
# =====================================================
if task == "regression":
    print("\n=== Regression Metrics ===")

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"MAE:  {mae:.4f}")
    print(f"MSE:  {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2:   {r2:.4f}")

    from sklearn.metrics import PredictionErrorDisplay

    fig, ax = plt.subplots(1, 2, figsize=(12,5))

    # Actual vs Predicted
    PredictionErrorDisplay.from_estimator(
        grid,
        X_test,
        y_test,
        kind="actual_vs_predicted",
        ax=ax[0]
    )
    ax[0].set_title("Actual vs Predicted")

    # Residual Plot
    PredictionErrorDisplay.from_estimator(
        grid,
        X_test,
        y_test,
        kind="residual_vs_predicted",
        ax=ax[1]
    )
    ax[1].set_title("Residuals vs Predicted")

    plt.tight_layout()
    plt.show()



# alphas = [0.01, 0.1, 1, 10, 100]
# coefs = []
#
# # Assuming X_train_scaled is already preprocessed
# for a in alphas:
#     model = Ridge(alpha=a) # Swap to Lasso() if needed
#     model.fit(X_train_scaled, y_train)
#     coefs.append(model.coef_)
#
# plt.plot(alphas, coefs)
# plt.xscale('log') # Crucial: makes the x-axis readable
# plt.title("Coefficient Path")
# plt.show()

# =====================================================
# 11. DECISION TREE VISUALIZATION (FOR EXP 6)
# =====================================================
# from sklearn.tree import plot_tree
# trained_tree = grid.best_estimator_[-1]
# plt.figure(figsize=(15, 10))
# plot_tree(trained_tree, filled=True, max_depth=3, rounded=True)
# plt.show()

# =====================================================
# 12. SVM DECISION BOUNDARY (FOR EXP 4)
# =====================================================
# from sklearn.inspection import DecisionBoundaryDisplay
# X_2d = preprocessor.fit_transform(X_train)[:, :2] # Grab only first 2 scaled features
# svc_2d = SVC(kernel='rbf').fit(X_2d, y_train)
#
# DecisionBoundaryDisplay.from_estimator(svc_2d, X_2d, response_method="predict", alpha=0.5)
# plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_train, edgecolors='k')
# plt.title("SVM Decision Boundary")
# plt.show()