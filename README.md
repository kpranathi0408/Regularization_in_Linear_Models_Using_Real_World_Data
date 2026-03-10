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
1. Data Preprocessing  
Steps performed:  
Load dataset using pandas  
Handle missing values (total_bedrooms)  
Split dataset into 80% training and 20% testing  
Standardize features using mean and standard deviation  

Example preprocessing:
df["total_bedrooms"].fillna(df["total_bedrooms"].median(), inplace=True)  
Feature scaling ensures all variables contribute equally to the model.  

