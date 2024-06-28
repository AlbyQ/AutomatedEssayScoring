import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error
import lightgbm as lgb

train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')

vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
X_train = vectorizer.fit_transform(train_data['full_text']).toarray()
X_test = vectorizer.transform(test_data['full_text']).toarray()
y_train = train_data['score']

assert X_train.shape[0] > 0, "Train data is empty"
assert X_test.shape[0] > 0, "Test data is empty"

X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

train_data_lgb = lgb.Dataset(X_tr, label=y_tr)
val_data_lgb = lgb.Dataset(X_val, label=y_val, reference=train_data_lgb)

params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'rmse',
    'num_leaves': 64,
    'learning_rate': 0.01,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'device': 'gpu'
}

print("Training model...")
gbm = lgb.train(params,
                train_data_lgb,
                num_boost_round=5000,
                valid_sets=[train_data_lgb, val_data_lgb])

y_val_pred = gbm.predict(X_val, num_iteration=gbm.best_iteration)
val_mse = mean_squared_error(y_val, y_val_pred)
val_rmse = np.sqrt(val_mse)

print('Validation RMSE:', val_rmse)

y_test_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)

y_test_pred_rounded = np.round(y_test_pred).astype(int)

submission = pd.DataFrame({
    'essay_id': test_data['essay_id'],
    'score': y_test_pred_rounded
})
submission.to_csv('submission.csv', index=False)
