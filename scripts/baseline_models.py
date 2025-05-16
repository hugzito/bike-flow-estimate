import pickle
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, mean_absolute_error

def load_graph(graph_num):
    with open(f'../data/graphs/{graph_num}/linegraph_tg.pkl', 'rb') as f:
        return pickle.load(f)

def split_data(x, y):
    X_train, X_temp, y_train, y_temp = train_test_split(x, y, test_size=0.3, random_state=100)
    X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=100)
    return X_train, X_valid, X_test, y_train, y_valid, y_test

def regression_task(x, y, x_bc):
    print('--- Regression Task ---')
    X_train, X_valid, X_test, y_train, y_valid, y_test = split_data(x, y)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_train_pred, y_valid_pred = model.predict(X_train), model.predict(X_valid)
    print('All Features:')
    print(f'Train R2: {r2_score(y_train, y_train_pred):.4f}, Valid R2: {r2_score(y_valid, y_valid_pred):.4f}')
    print(f'Train MAE: {mean_absolute_error(y_train, y_train_pred):.2f}, Valid MAE: {mean_absolute_error(y_valid, y_valid_pred):.2f}')

    X_train_bc, X_valid_bc, X_test_bc, y_train_bc, y_valid_bc, y_test_bc = split_data(x_bc, y)
    model_bc = LinearRegression()
    model_bc.fit(X_train_bc, y_train_bc)
    y_train_bc_pred, y_valid_bc_pred = model_bc.predict(X_train_bc), model_bc.predict(X_valid_bc)
    print('\nBC Feature Only:')
    print(f'Train R2: {r2_score(y_train_bc, y_train_bc_pred):.4f}, Valid R2: {r2_score(y_valid_bc, y_valid_bc_pred):.4f}')
    print(f'Train MAE: {mean_absolute_error(y_train_bc, y_train_bc_pred):.2f}, Valid MAE: {mean_absolute_error(y_valid_bc, y_valid_bc_pred):.2f}')

def classification_task(x, y, x_bc, task_name='Classification'):
    print(f'\n--- {task_name} ---')
    X_train, X_valid, X_test, y_train, y_valid, y_test = split_data(x, y)
    model = LogisticRegression(max_iter=int(1e5), solver='saga', random_state=100, multi_class='auto')
    model.fit(X_train, y_train)
    print('All Features:')
    print(f'Valid Accuracy: {model.score(X_valid, y_valid):.4f}')

    X_train_bc, X_valid_bc, X_test_bc, y_train_bc, y_valid_bc, y_test_bc = split_data(x_bc, y)
    model_bc = LogisticRegression(max_iter=int(1e5), solver='saga', random_state=100, multi_class='auto')
    model_bc.fit(X_train_bc, y_train_bc)
    print('\nBC Feature Only:')
    print(f'Valid Accuracy: {model_bc.score(X_valid_bc, y_valid_bc):.4f}')

    print('\nOn Test Set:')
    print(f'All Features Test Accuracy: {model.score(X_test, y_test):.4f}')
    print(f'BC Feature Only Test Accuracy: {model_bc.score(X_test_bc, y_test_bc):.4f}')

def main(graph_num=26):
    graph_data = load_graph(graph_num)

    y_raw = graph_data.y[graph_data.y > 0]
    x_raw = graph_data.x[graph_data.y > 0].numpy()
    x_bc = x_raw[:, 5].reshape(-1, 1)

    # Regression task
    regression_task(x_raw, y_raw.numpy(), x_bc)

    # Binary classification
    y_bin = torch.bucketize(y_raw, boundaries=torch.tensor([3000])).numpy()
    classification_task(x_raw, y_bin, x_bc, 'Binary Classification')

    # Multi-class classification
    bins_10 = torch.tensor([400, 800, 1300, 2100, 3000, 3700, 4700, 7020, 9660])
    y_multi = torch.bucketize(y_raw, boundaries=bins_10).numpy()
    classification_task(x_raw, y_multi, x_bc, 'Multi-Class Classification')

if __name__ == "__main__":
    main()
