import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# ======================
# LOAD DATA
# ======================
df = pd.read_csv(r"C:\24EG107F25\tips.csv")

print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())

# ======================
# EDA VISUALIZATION
# ======================
sns.histplot(df['total_bill'], kde=True)
plt.title("Total Bill Distribution")
plt.show()

sns.boxplot(x=df['total_bill'])
plt.title("Boxplot - Outlier Detection")
plt.show()

sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# ======================
# OUTLIER REMOVAL
# ======================
for col in ['total_bill', 'tip']:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df[col] >= Q1 - 1.5*IQR) & (df[col] <= Q3 + 1.5*IQR)]

df = df.drop(['day', 'time'], axis=1)
# Encoding
df = pd.get_dummies(df, drop_first=True)

# ======================
# SPLIT
# ======================
X = df.drop('tip', axis=1)
y = df['tip']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ======================
# MODELS
# ======================
lr = LinearRegression()
rf = RandomForestRegressor()
xgb = XGBRegressor()

lr.fit(X_train, y_train)
rf.fit(X_train, y_train)
xgb.fit(X_train, y_train)

# ======================
# EVALUATION
# ======================
results = []

for name, model in [("Linear Regression", lr), ("Random Forest", rf), ("XGBoost", xgb)]:
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    results.append([name, r2, rmse])

results_df = pd.DataFrame(results, columns=["Model", "R2", "RMSE"])
print(results_df)

# CV Score (RF)
cv = cross_val_score(rf, X, y, cv=5)
print("CV Score (RF):", cv.mean())

# ======================
# FEATURE IMPORTANCE
# ======================
plt.figure(figsize=(8,5))
plt.barh(X.columns, rf.feature_importances_)
plt.title("Feature Importance (Random Forest)")
plt.show()

# ======================
# SAVE
# ======================
pickle.dump(lr, open('lr.pkl', 'wb'))
pickle.dump(rf, open('rf.pkl', 'wb'))
pickle.dump(xgb, open('xgb.pkl', 'wb'))
pickle.dump(X.columns, open('columns.pkl', 'wb'))
pickle.dump(results_df, open('results.pkl', 'wb'))

print("All models saved!")
