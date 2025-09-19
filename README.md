# Linear and Logistic Regression Models

## Overview

This project demonstrates the application of **Linear Regression** and **Logistic Regression** using Python’s **scikit-learn** library.  
The work was completed as part of a **Machine Learning course assignment** and showcases practical regression and classification methods with real datasets.

---

## Table of Contents
- [Project Description](#project-description)
- [Datasets](#datasets)
- [Preprocessing Steps](#preprocessing-steps)
- [Methodology](#methodology)
- [Results](#results)
- [Technologies Used](#technologies-used)

---

## Project Description

The assignment was divided into three main tasks:

1. **Basic Linear Regression**  
   - Predict restaurant profits based on the population of the city.  
   - Fit a regression line and use it to predict profit for a city with 18 habitants.  

2. **Binary Logistic Regression**  
   - Predict whether a job applicant will be hired based on two exam scores.  
   - Train a logistic regression model, visualize predictions, and analyze misclassified points.  

3. **Multi-class Classification (Theory)**  
   - Explain how **One-vs-Rest (OvR)** and **One-vs-One (OvO)** strategies extend logistic regression to multi-class problems.  

---

## Datasets

1. **RegressionData.csv**  
   - Columns:  
     - `X` → City population (in thousands).  
     - `y` → Profit/loss of restaurant (in $10,000 units).  

2. **LogisticRegressionData.csv**  
   - Columns:  
     - `Score1` → First technical interview score.  
     - `Score2` → Second technical interview score.  
     - `y` → Hiring decision (`0 = rejected`, `1 = hired`).  

---

## Preprocessing Steps

1. Loaded datasets with **pandas**.  
2. Reshaped features for scikit-learn models.  
3. Visualized data using **Matplotlib** scatter plots.  
4. Trained models using **scikit-learn regression classes**.  

---

## Methodology

1. **Linear Regression**  
   - Used `LinearRegression` from scikit-learn.  
   - Modeled relationship as:  
     \[
     y = b_0 + b_1X
     \]  
   - Predicted profit for city population = 18.  

2. **Logistic Regression**  
   - Used `LogisticRegression` from scikit-learn.  
   - Trained classifier on applicant exam scores.  
   - Visualized classification boundaries and training errors.  

3. **Multi-class Classification (Theory)**  
   - **OvR** → Train one classifier per class vs. all others.  
   - **OvO** → Train one classifier for each pair of classes; majority vote decides the class.  

---

## Results

- **Linear Regression**  
  - Fitted line clearly modeled the relationship between population and profit.  
  - Successfully predicted profit for city population = 18.  

- **Logistic Regression**  
  - Classifier separated hired vs. rejected applicants based on scores.  
  - Visualizations revealed some misclassifications (expected in real-world data).  

- **Multi-class Classification**  
  - Theoretical understanding of OvR and OvO approaches was explained in the code.  

---

## Technologies Used

- **Languages**: Python 3  
- **Libraries**:  
  - Pandas  
  - NumPy  
  - Matplotlib  
  - Scikit-learn  



