import pickle

import cupy as cp
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')

# train_data = train_data.head(100)
# test_data = test_data.head(100)

vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
X_train = vectorizer.fit_transform(train_data['full_text']).toarray()
X_test = vectorizer.transform(test_data['full_text']).toarray()
y_train = train_data['score']

assert X_train.shape[0] > 0, "Train data is empty"
assert X_test.shape[0] > 0, "Test data is empty"

# X_train = cp.array(X_train)
# X_test = cp.array(X_test)

X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

xgb = XGBRegressor(tree_method='hist', random_state=42)

param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [6, 8, 10],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}

random_search = RandomizedSearchCV(xgb, param_distributions=param_dist, n_iter=20, scoring='neg_mean_squared_error',
                                   cv=5, verbose=1, random_state=42, error_score='raise')
random_search.fit(X_tr, y_tr)

best_xgb = random_search.best_estimator_

model_filename = 'model/best_xgb_model.pkl'
with open(model_filename, 'wb') as file:
    pickle.dump(best_xgb, file)

y_val_pred = best_xgb.predict(X_val)
val_mse = mean_squared_error(y_val, y_val_pred)
val_rmse = np.sqrt(val_mse)

y_test_pred = best_xgb.predict(X_test)
y_test_pred_rounded = np.round(y_test_pred).astype(int)

print('Validation RMSE:', val_rmse)

submission = pd.DataFrame({
    'essay_id': test_data['essay_id'],
    'score': y_test_pred_rounded
})
submission.to_csv('result/submission.csv', index=False)
