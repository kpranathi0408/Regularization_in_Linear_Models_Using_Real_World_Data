# Assignment On Linear Models
# Linear Model Using Real-world data

# California Housing Prices
# Data: url : https://www.kaggle.com/datasets/camnugent/california-housing-prices?resource

# Importing 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet

# using pandas to frame
df = pd.read_csv("ass_linearmodelusingrealwrlddata(housing).csv")
print(df.head())

print(df.columns)

df["total_bedrooms"].fillna(df["total_bedrooms"].median(), inplace = True)

print(df.isnull().sum())

X = df.drop(columns = ["median_house_value"])
y = df["median_house_value"]

# converting into np arrary
X = np.array(X)
y = np.array(y)

# Train–Test Split
np.random.seed(42)

indices = np.arange(X.shape[0])
np.random.shuffle(indices)

split = int(0.8 * X.shape[0])

train_idx = indices[:split]
test_idx = indices[split:]

X_train = X[train_idx]
X_test = X[test_idx]
y_train = y[train_idx]
y_test = y[test_idx]


mean = X_train.mean(axis = 0)
std = X_train.std(axis = 0)

X_train_scaled = (X_train - mean) / std
X_test_scaled = (X_test - mean) / std


# Multiple Linear Regression

model = LinearRegression()
model.fit(X_train_scaled, y_train)

train_pred = model.predict(X_train_scaled)
test_pred = model.predict(X_test_scaled)

print("Linear Regression Train MSE:", mean_squared_error(y_train, train_pred))
print("Linear Regression Test MSE:", mean_squared_error(y_test, test_pred))

# Ridge

from sklearn.linear_model import Ridge

alphas = [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]

ridge_train_errors = []
ridge_test_errors = []

for a in alphas:
    model = Ridge(alpha = a)
    model.fit(X_train_scaled, y_train)

    ridge_train_errors.append(
        mean_squared_error(y_train, model.predict(X_train_scaled))
    )
    ridge_test_errors.append(
        mean_squared_error(y_test, model.predict(X_test_scaled))
    )

print("Ridge_train_errors: ",ridge_train_errors)
print("Ridge_test_errors: ",ridge_test_errors)


# Lasso
from sklearn.linear_model import Lasso

lasso_train_errors = []
lasso_test_errors = []

for a in alphas:
    model = Lasso(alpha = a, max_iter = 10000)
    model.fit(X_train_scaled, y_train)

    lasso_train_errors.append(
        mean_squared_error(y_train, model.predict(X_train_scaled))
    )
    lasso_test_errors.append(
        mean_squared_error(y_test, model.predict(X_test_scaled))
    )

print("Lasso_train_errors: " ,lasso_train_errors)
print("Lasso_test_errors: " ,lasso_test_errors)

# ElasticNet 

from sklearn.linear_model import ElasticNet

elastic_train_errors = []
elastic_test_errors = []

for a in alphas:
    model = ElasticNet(alpha = a, l1_ratio = 0.5, max_iter = 10000)
    model.fit(X_train_scaled, y_train)

    elastic_train_errors.append(
        mean_squared_error(y_train, model.predict(X_train_scaled))
    )
    elastic_test_errors.append(
        mean_squared_error(y_test, model.predict(X_test_scaled))
    )

print("Elastic_train_errors: ",elastic_train_errors)
print("Elastic_test_errors: ",elastic_test_errors)


# ---------------Plots-------------
# Ridge: Train vs Test Error Plot

plt.figure(figsize = (8, 5))
plt.plot(alphas, ridge_train_errors, marker = '*', label = 'Ridge Train MSE', color = 'yellow')
plt.plot(alphas, ridge_test_errors, marker = '*', label = 'Ridge Test MSE', color = 'green')

plt.xscale('log')
plt.xlabel('Alpha (log scale)')
plt.ylabel('Mean Squared Error')
plt.title('Ridge Regression: Train vs Test Error')
plt.legend()
plt.grid(True, alpha = 0.3)
plt.show()

# Lasso: Train vs Test Error Plot

plt.figure(figsize = (8, 5))
plt.plot(alphas, lasso_train_errors, marker = '*', label = 'Lasso Train MSE',color = 'black' )
plt.plot(alphas, lasso_test_errors, marker = '*', label = 'Lasso Test MSE', color = 'skyblue')

plt.xscale('log')
plt.xlabel('Alpha (log scale)')
plt.ylabel('Mean Squared Error')
plt.title('Lasso Regression: Train vs Test Error')
plt.legend()
plt.grid(True)
plt.show()

# Elastic Net: Train vs Test Error Plot

plt.figure(figsize = (8, 5))
plt.plot(alphas, elastic_train_errors, marker = '*', label = 'Elastic Net Train MSE')
plt.plot(alphas, elastic_test_errors, marker = '*', label = 'Elastic Net Test MSE')

plt.xscale('log')
plt.xlabel('Alpha (log scale)')
plt.ylabel('Mean Squared Error')
plt.title('Elastic Net: Train vs Test Error')
plt.legend()
plt.grid(True)
plt.show()

# Ridge: Coefficient Path

ridge_coefs = []

for a in alphas:
    model = Ridge(alpha = a)
    model.fit(X_train_scaled, y_train)
    ridge_coefs.append(model.coef_)

ridge_coefs = np.array(ridge_coefs)

plt.figure(figsize = (8, 5))
for i in range(ridge_coefs.shape[1]):
    plt.plot(alphas, ridge_coefs[:, i], marker = '*')

plt.xscale('log')
plt.xlabel('Alpha (log scale)')
plt.ylabel('Coefficient Value')
plt.title('Ridge Coefficient Shrinkage Path')
plt.grid(True)
plt.show()

# Lasso: Coefficient Path

lasso_coefs = []

for a in alphas:
    model = Lasso(alpha = a, max_iter = 10000)
    model.fit(X_train_scaled, y_train)
    lasso_coefs.append(model.coef_)

lasso_coefs = np.array(lasso_coefs)

plt.figure(figsize=(8, 5))
for i in range(lasso_coefs.shape[1]):
    plt.plot(alphas, lasso_coefs[:, i], marker = '*')

plt.xscale('log')
plt.xlabel('Alpha (log scale)')
plt.ylabel('Coefficient Value')
plt.title('Lasso Coefficient Path')
plt.grid(True)
plt.show()

# Elastic Net: Coefficient Path

elastic_coefs = []

for a in alphas:
    model = ElasticNet(alpha = a, l1_ratio = 0.5, max_iter = 10000)
    model.fit(X_train_scaled, y_train)
    elastic_coefs.append(model.coef_)

elastic_coefs = np.array(elastic_coefs)

plt.figure(figsize=(8, 5))
for i in range(elastic_coefs.shape[1]):
    plt.plot(alphas, elastic_coefs[:, i], marker = '*')

plt.xscale('log')
plt.xlabel('Alpha (log scale)')
plt.ylabel('Coefficient Value')
plt.title('Elastic Net Coefficient Path')
plt.grid(True)
plt.show()

print("Successfuly completed the assignment")


# -------------------- Loss Function -------------------

# Multiple Linear Reg :
# --------------- L = n1 ​∑(y−y^​) 2
# Only minimizes error
# No complexity control
# High variance risk

# Linear regression minimizes only the squared error between predictions and actual values, so it focuses
#  purely on fitting the training data without controlling model complexity

# Ridge Reg : 
# --------------- L = MSE + λ ∑w2
# Penalizes large weights
# Shrinks all coefficients
# Reduces variance

# Ridge regression adds a penalty on the squared magnitude of coefficients, which discourages large weights 
# and reduces overfitting while keeping all features.

# Lasso Reg :
#  --------------- L = MSE + λ × Σ|β|
# Encourages sparsity
# Sets some weights to zero
# Performs feature selection

# Lasso regression adds a penalty on the absolute values of coefficients, which forces some weights to 
# become exactly zero and effectively removes less important features.

# Elastic Net Reg :
# --------------- L = MSE + λ(α ∑∣w∣ + (1−α) ∑w ^ 2)
# Combines stability + sparsity
# Handles correlated features better than Lasso

# Elastic Net combines both L1 and L2 penalties, allowing the model to both shrink coefficients and 
# eliminate some features depending on their importance.

# --------------------------- Analysis & Hypothesis ------------------------------

# 1. Why does regularization improve test performance?

# Regularization improves test performance because it prevents the model from overfitting.
# Without regularization, linear regression tries too hard to fit the training data and ends up
# learning noise. When we add regularization, we slightly relax the fit on training data, which
# reduces variance. As a result, the model generalizes better to unseen data, which is why test error goes down.

# 2. Why does Ridge keep all features but shrink them?

# Ridge uses an L2 penalty, which discourages large coefficients but does not push them to zero.
# So instead of removing features, Ridge reduces their influence by shrinking their weights. 
# This is especially useful when features are correlated, because Ridge distributes the weight among them instead 
# of choosing one and discarding others.

# 3. Why does Elastic Net behave differently from both Ridge and Lasso?

# Elastic Net combines both L1 and L2 penalties.
# Because of this, it shows both behaviors: some coefficients are shrunk smoothly like Ridge, while others
# are eliminated like Lasso. In my experiment, the combined penalty became too strong at higher values of alpha,
# which restricted the model too much and caused underfitting.

# 4. Which model performed best on your dataset and why?

# Ridge Regression performed the best on my dataset.
# The California Housing dataset has multiple informative and correlated features, and Ridge handles this well by shrinking coefficients without removing useful predictors. Lasso removed some features unnecessarily, and Elastic Net over-regularized the model. That’s why Ridge achieved the lowest test error and best generalization.
