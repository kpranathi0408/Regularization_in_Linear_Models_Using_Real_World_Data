# Regularization_in_Linear_Models

# Assignment Overview

This assignment focuses on understanding how regularization techniques control overfitting in regression models.

Using a real-world dataset (California Housing Prices), the following models are implemented and compared:  
Multiple Linear Regression  
Ridge Regression  
Lasso Regression  
Elastic Net Regression  
The goal is to study how different regularization penalties influence model complexity, coefficient behavior, and generalization performance.  

# Source code: 
ass_linearmodelusingrealwrlddata  

# Dataset

Dataset: California Housing Prices  
Source: Kaggle  

* Target variable: *
median_house_value

# Features include variables such as:
median_income  
housing_median_age  
total_rooms  
total_bedrooms  
population  
households 
latitude  
longitude  
The dataset represents housing characteristics in California districts.  

# Objectives
The assignment aims to understand:  
How regularization reduces overfitting.  
How different penalties modify the loss function.  
How coefficients behave under different regularization strengths.  
Which model generalizes best on real-world data.   
All conclusions are supported using error analysis and coefficient path visualizations.  

# Methodology  

# Experiments and Visualizations

The assignment generates several plots.
1. Training vs Testing Error
   
For each model: 
Ridge Regression
Lasso Regression
Elastic Net

These plots show how regularization strength affects model performance.

2. Coefficient Shrinkage Paths

Coefficient path plots illustrate how feature weights change as alpha increases.

Observations: 
Ridge gradually shrinks coefficients
Lasso pushes some coefficients to zero
Elastic Net shows mixed behavior

# Analysis
Why does regularization improve test performance?
Regularization prevents the model from fitting noise in the training data.

Examples:
Linear regression may overfit when many features exist.
Adding penalties forces the model to prefer simpler solutions.

Why does Ridge shrink weights but keep features?
The L2 penalty discourages large weights but does not force them to zero.

Examples:
When predictors are correlated
When all variables contain useful information
Why does Lasso perform feature selection?
The L1 penalty pushes small coefficients to exactly zero.

Examples:
Removing weak predictors
Creating simpler interpretable models

Why does Elastic Net behave differently?
Elastic Net combines both L1 and L2 penalties.

Examples:
Some coefficients shrink like Ridge
Some features are eliminated like Lasso

# Results

Observations from the experiment:  
Linear regression shows higher risk of overfitting.
Ridge regression stabilizes coefficients and improves generalization.
Lasso removes less important predictors.
Elastic Net balances both shrinkage and sparsity.

Best performing model: Ridge Regression
Reason: The dataset contains multiple correlated features, and Ridge handles correlation effectively without discarding useful variables.

