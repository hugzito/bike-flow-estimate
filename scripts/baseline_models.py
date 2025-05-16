# %%
import pandas as pd
import numpy as np
import os
import sys
from sklearn.linear_model import LogisticRegression, LinearRegression
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, r2_score, mean_absolute_error
import pickle
import torch

graph_num = 26

with open(f'../data/graphs/{graph_num}/linegraph_tg.pkl', 'rb') as f:
    graph_data = pickle.load(f)

bins_10 = torch.tensor([int(i) for i in '400 800 1300 2100 3000 3700 4700 7020 9660'.split()])
bins_binary = torch.tensor([int(i) for i in '3000'.split()])


# %%
y = graph_data.y[graph_data.y > 0]
x = graph_data.x[graph_data.y > 0].numpy()

x_bc = x[:, 5].reshape(-1, 1)

print(f'X shape: {x.shape}')


# %%
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=100)
X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=100)
regression_model = LinearRegression()

regression_model.fit(X_train, y_train)
y_train_pred = regression_model.predict(X_train)
y_valid_pred = regression_model.predict(X_valid)

print('With All Features:')
print('Train R2:', r2_score(y_train, y_train_pred))
print('Valid R2:', r2_score(y_valid, y_valid_pred))
print('Train MAE:', mean_absolute_error(y_train, y_train_pred))
print('Valid MAE:', mean_absolute_error(y_valid, y_valid_pred))

print('\n\n')

print('With BC Feature:')
X_train_bc, X_test_bc, y_train_bc, y_test_bc = train_test_split(x_bc, y, test_size=0.3, random_state=100)
X_valid_bc, X_test_bc, y_valid_bc, y_test_bc = train_test_split(X_test_bc, y_test_bc, test_size=0.5, random_state=100)
regression_model_bc = LinearRegression()
regression_model_bc.fit(X_train_bc, y_train_bc)
y_train_bc_pred = regression_model_bc.predict(X_train_bc)
y_valid_bc_pred = regression_model_bc.predict(X_valid_bc)
print('Train R2:', r2_score(y_train_bc, y_train_bc_pred))
print('Valid R2:', r2_score(y_valid_bc, y_valid_bc_pred))
print('Train MAE:', mean_absolute_error(y_train_bc, y_train_bc_pred))
print('Valid MAE:', mean_absolute_error(y_valid_bc, y_valid_bc_pred))


# %%
y = graph_data.y[graph_data.y > 0]
x = graph_data.x[graph_data.y > 0].numpy()
y = torch.bucketize(y, boundaries=bins_binary).numpy()


# %%
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=100)
X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=100)


# %%
print('ON BINARY CLASSIFICATION TASK')
print('With All Features:')
# Create a Logistic Regression model
model = LogisticRegression(max_iter=int(1e+5), solver='saga', random_state=100, multi_class='auto')
# Fit the model on the training data
model.fit(X_train, y_train)
# Evaluate the model on the validation data
valid_accuracy = model.score(X_valid, y_valid)

print(f"Validation accuracy: {valid_accuracy:.4f}")

# print(classification_report(y_valid, model.predict(X_valid), zero_division=0))

print('\n\n')

print('With BC :')
X_train_bc, X_test_bc, y_train_bc, y_test_bc = train_test_split(x_bc, y, test_size=0.3, random_state=100)
X_valid_bc, X_test_bc, y_valid_bc, y_test_bc = train_test_split(X_test_bc, y_test_bc, test_size=0.5, random_state=100)
# Create a Logistic Regression model
model_bc = LogisticRegression(max_iter=int(1e+5), solver='saga', random_state=100, multi_class='auto')
# Fit the model on the training data
model_bc.fit(X_train_bc, y_train_bc)
# Evaluate the model on the validation data
valid_accuracy_bc = model_bc.score(X_valid_bc, y_valid_bc)
print(f"Validation accuracy: {valid_accuracy_bc:.4f}")
# print(classification_report(y_valid_bc, model_bc.predict(X_valid_bc), zero_division=0))


# %%
y = graph_data.y[graph_data.y > 0]
y = torch.bucketize(y, boundaries=bins_10).numpy()
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=100)
X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=100)

print('ON MULTI CLASS CLASSIFICATION TASK')
print('With All Features:')
# Create a Logistic Regression model
model = LogisticRegression(max_iter=int(1e+5), solver='saga', random_state=100, multi_class='auto')
# Fit the model on the training data
model.fit(X_train, y_train)
# Evaluate the model on the validation data
valid_accuracy = model.score(X_valid, y_valid)
print(f"Validation accuracy: {valid_accuracy:.4f}")
# print(classification_report(y_valid, model.predict(X_valid), zero_division=0))
print('\n\n')
print('With BC :')
X_train_bc, X_test_bc, y_train_bc, y_test_bc = train_test_split(x_bc, y, test_size=0.3, random_state=100)
X_valid_bc, X_test_bc, y_valid_bc, y_test_bc = train_test_split(X_test_bc, y_test_bc, test_size=0.5, random_state=100)
# Create a Logistic Regression model
model_bc = LogisticRegression(max_iter=int(1e+5), solver='saga', random_state=100, multi_class='auto')
# Fit the model on the training data
model_bc.fit(X_train_bc, y_train_bc)
# Evaluate the model on the validation data
valid_accuracy_bc = model_bc.score(X_valid_bc, y_valid_bc)
print(f"Validation accuracy: {valid_accuracy_bc:.4f}")
# print(classification_report(y_valid_bc, model_bc.predict(X_valid_bc), zero_division=0))


# %%
### show performance on test set
print('ON TEST SET')
print('With All Features:')
# Create a Logistic Regression model
model = LogisticRegression(max_iter=int(1e+5), solver='saga', random_state=100, multi_class='auto')
# Fit the model on the training data
model.fit(X_train, y_train)
# Evaluate the model on the validation data
test_accuracy = model.score(X_test, y_test)
print(f"Test accuracy: {test_accuracy:.4f}")
# print(classification_report(y_test, model.predict(X_test), zero_division=0))
print('\n\n')
print('With BC :')
X_train_bc, X_test_bc, y_train_bc, y_test_bc = train_test_split(x_bc, y, test_size=0.3, random_state=100)
X_valid_bc, X_test_bc, y_valid_bc, y_test_bc = train_test_split(X_test_bc, y_test_bc, test_size=0.5, random_state=100)
# Create a Logistic Regression model
model_bc = LogisticRegression(max_iter=int(1e+5), solver='saga', random_state=100, multi_class='auto')
# Fit the model on the training data
model_bc.fit(X_train_bc, y_train_bc)
# Evaluate the model on the validation data
test_accuracy_bc = model_bc.score(X_test_bc, y_test_bc)
print(f"Test accuracy: {test_accuracy_bc:.4f}")
# print(classification_report(y_test_bc, model_bc.predict(X_test_bc), zero_division=0))


# %%



