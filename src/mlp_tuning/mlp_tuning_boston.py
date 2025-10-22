import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, mean_squared_error
# import MLPRegressor để sử dụng cho phần tuning MLPRegressor
from sklearn.neural_network import MLPRegressor
# import MLPClassifier để sử dụng cho phần tuning MLPClassifier
from sklearn.neural_network import MLPClassifier


# Thực hiện tuning params cho thuật toán MLPRegressor/MLPClassifier trên bộ dữ liệu Boston Housing dataset.
def bostonHousingData():
    df = pd.read_csv(r"..\..\resource\BostonHousing.csv")
    X = df.iloc[:, :-1]  # all columns except target
    y = df.iloc[:, -1]  # target (label)
    return X, y

# MLPRegressor
def mlp_tuning_boston_MLPRegressor():
    X, y = bostonHousingData()

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPRegressor(max_iter=1000, random_state=1))
    ])

    param_grid = {
        'mlp__hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
        'mlp__activation': ['relu', 'tanh'],
        'mlp__alpha': [0.0001, 0.001, 0.01],
        'mlp__learning_rate_init': [0.001, 0.01, 0.1]
    }

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=make_scorer(mean_squared_error, greater_is_better=False),
        cv=5,
        n_jobs=-1,
        verbose=2
    )

    grid_search.fit(X, y)

    print("Best parameters found: ", grid_search.best_params_)
    best_rmse = np.sqrt(-grid_search.best_score_)   
    print("Best RMSE: ", best_rmse)

# MLPClassifier
def mlp_tuning_boston_MLPClassifier():
    X, y = bostonHousingData()
    y = (y > y.median()).astype(int)  # Chuyển đổi thành bài toán phân loại nhị phân

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPClassifier(max_iter=1000, random_state=1))
    ])

    param_grid = {
        'mlp__hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
        'mlp__activation': ['relu', 'tanh'],
        'mlp__alpha': [0.0001, 0.001, 0.01],
        'mlp__learning_rate_init': [0.001, 0.01, 0.1]
    }

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring='accuracy',
        cv=5,
        n_jobs=-1,
        verbose=2
    )

    grid_search.fit(X, y)
    print("Best parameters found: ", grid_search.best_params_)
    print("Best accuracy: ", grid_search.best_score_)

# Chạy hàm tuning
if __name__ == "__main__":
    print("Tuning MLPRegressor on Boston Housing dataset:")
    mlp_tuning_boston_MLPRegressor()
    print("\nTuning MLPClassifier on Boston Housing dataset:")
    mlp_tuning_boston_MLPClassifier()

    