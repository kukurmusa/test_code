import optuna
import lightgbm as lgb
from sklearn.metrics import mean_squared_error

# Create dataset (you already have X_train, X_test, y_train, y_test)
train_data = lgb.Dataset(X_train, label=y_train)
valid_data = lgb.Dataset(X_test, label=y_test)

# Objective function for optuna
def objective(trial):
    param = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'verbosity': -1,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'num_leaves': trial.suggest_int('num_leaves', 16, 128),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
        'lambda_l1': trial.suggest_float('lambda_l1', 0, 5),
        'lambda_l2': trial.suggest_float('lambda_l2', 0, 5)
    }
    
    gbm = lgb.train(param, train_data, valid_sets=[valid_data], 
                    num_boost_round=100, early_stopping_rounds=10, verbose_eval=False)
    
    preds = gbm.predict(X_test)
    return mean_squared_error(y_test, preds, squared=False)  # RMSE

# Run optimisation
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

# Best params
print("Best trial:")
print(f"  RMSE: {study.best_value:.4f}")
print(f"  Params: {study.best_params}")


best_params = study.best_params
best_params.update({
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'verbosity': -1
})

final_model = lgb.train(
    best_params,
    train_data,
    valid_sets=[valid_data],
    num_boost_round=200,
    early_stopping_rounds=20
)

# Final predictions
y_pred = final_model.predict(X_test)
