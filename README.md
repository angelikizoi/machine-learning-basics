# Machine Learning and Data Analysis Basics

Welcome to the **Machine Learning and Data Analysis Basics** repository. This repository contains a collection of Jupyter notebooks and Python scripts that cover fundamental concepts in machine learning and data analysis using various Python libraries such as `scipy` and `scikit-learn`.

## Table of Contents

1. [Scipy T-tests](#scipy-t-tests)
2. [Scipy Optimization Algorithms](#scipy-optimization-algorithms)
3. [Linear Regression](#linear-regression)
    - Scipy Implementation
    - Scikit-learn Implementation
4. [Logistic Regression](#logistic-regression)
    - Custom Logistic Regression Class
    - Scikit-learn Implementation
5. [Naive Bayes (scikit-learn)](#naive-bayes-scikit-learn)
6. [Decision Trees (scikit-learn)](#decision-trees-scikit-learn)
7. [K-Means Clustering](#k-means-clustering)
8. [Principal Component Analysis (PCA) with K-Means](#principal-component-analysis-pca-with-k-means)
9. [K-Fold Cross-Validation](#k-fold-cross-validation)
10. [Complete Example](#complete-example)
    - Data Cleansing
    - Feature Engineering (Dummy Variables, Feature Selection, Feature Extraction)
    - Feature Scaling (MinMaxScaler)
    - Hyperparameter Tuning (GridSearchCV-Random Forest)
    - Feature Importance

---

## 1. Scipy T-tests

This notebook demonstrates the use of **T-tests** to check if two independent samples have different means, and to check for normality of distributions using the `scipy.stats` library.

- **File:** `1_ScipyBasics.ipynb`
- **Key Concepts:** Independent two-sample T-tests, Normality tests, Linear Equations

---

## 2. Scipy Optimization Algorithms

This notebook explores various **optimization algorithms** using the `scipy.optimize` module, including:
- **Linear Programming (LP)**
- **Task Assignment (Hungarian Algorithm)**
- **Euclidean Distances** for optimization problems.

- **File:** `2_Scipy_Optimization.ipynb`
- **Key Concepts:** Linear Programming, Task Assignment, Euclidean Distance Calculations

---

## 3. Linear Regression

### a. Scipy Implementation

This notebook demonstrates how to perform **linear regression** using `scipy`, including calculating the **R-squared value** and plotting the regression line.

- **File:** `3_LinearRegression.ipynb`
- **Key Concepts:** R-squared, Plotting results, Regression line

### b. Scikit-learn Implementation

This notebook covers **linear regression** using the `LinearRegression` model from the `scikit-learn` library, with model fitting, predictions, and evaluation.

- **File:** `3_LinearRegression.ipynb`
- **Key Concepts:** Model fitting, predictions, performance evaluation (R-squared)

---

## 4. Logistic Regression

### a. Custom Logistic Regression Class

This notebook contains a **custom implementation** of logistic regression from scratch, explaining the gradient descent, sigmoid function, and cost function.

- **File:** `4a_LogisticRegressionAlgorythm.ipynb`
- **Key Concepts:** Sigmoid function, Cost function, Gradient Descent

### b. Scikit-learn Implementation

This notebook demonstrates logistic regression using the `LogisticRegression` model from `scikit-learn` to solve binary classification problems.

- **File:** `4b_LogisticRegression.ipynb`
- **Key Concepts:** Binary classification, Regularization, Model evaluation

---

## 5. Naive Bayes (scikit-learn)

This notebook covers **Gaussian Naive Bayes classification** using `scikit-learn` and evaluates the model's performance with standard classification metrics.

- **File:** `5_NaiveBayes.ipynb`
- **Key Concepts:** Gaussian Naive Bayes, Model evaluation, Accuracy, Precision, Recall

---

## 6. Decision Trees (scikit-learn)

This notebook explores **Decision Trees** for classification using `scikit-learn`, highlighting the importance of tree depth, Gini impurity, and information gain.

- **File:** `6_DecisionTrees.ipynb`
- **Key Concepts:** Gini Index, Entropy, Decision Boundaries, Overfitting

---

## 7. K-Means Clustering

### a. K-Means with Random Data

This notebook applies **K-Means Clustering** on a randomly generated dataset to demonstrate the concept of clustering and centroid convergence.

- **File:** `7a_KMeansIntro.ipynb`
- **Key Concepts:** Centroid Initialization, Cluster Assignments, Elbow Method

### b. K-Means with Iris Dataset

This notebook applies **K-Means Clustering** to the famous Iris dataset, explaining how to identify clusters and use PCA for dimensionality reduction.

- **File:** `7_KMeans.ipynb`
- **Key Concepts:** Clustering, Elbow Method, Iris Dataset

---

## 8. Principal Component Analysis (PCA) with K-Means

This notebook combines **Principal Component Analysis (PCA)** with **K-Means Clustering** to reduce dimensionality before applying clustering on the Iris dataset.

- **File:** `8_PCA.ipynb`
- **Key Concepts:** PCA for Dimensionality Reduction, K-Means Clustering, Visualization

---

## 9. K-Fold Cross-Validation

This notebook covers **K-Fold Cross-Validation**, a technique used to evaluate machine learning models by splitting the training data into multiple folds and iterating through each fold as a validation set.

- **File:** `9_KFold.ipynb`
- **Key Concepts:** Cross-validation, Model evaluation, Train-test split, Generalization

---

## 10. Complete Example: Data Processing and Hyperparameter Tuning

This notebook showcases a complete machine learning pipeline, including:
- **Data Cleansing**: Handling missing values, correcting data types
- **Feature Engineering**: Dummy variables, feature selection, feature extraction
- **Feature Scaling**: Using MinMaxScaler
- **Hyperparameter Tuning**: Using `GridSearchCV` with Random Forest
- **Feature Importance**: Analyzing feature importance from the Random Forest model

- **File:** `10_RandomForestCompleted.ipynb`
- **Key Concepts:** End-to-end machine learning pipeline, GridSearchCV, Random Forest, Feature Scaling

---

