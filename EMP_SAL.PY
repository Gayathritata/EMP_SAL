import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

data = {
    'Experience': np.random.randint(1, 15, 100),
    'Education': np.random.choice(['Bachelors', 'Masters', 'PhD'], 100),
    'Job_Role': np.random.choice(['Engineer', 'Manager', 'Analyst'], 100),
    'Industry': np.random.choice(['IT', 'Finance', 'Healthcare'], 100)
}
df = pd.DataFrame(data)

def generate_salary(row):
    base = 30000 + row['Experience'] * 2500
    if row['Education'] == 'Masters': base += 8000
    elif row['Education'] == 'PhD': base += 15000
    if row['Job_Role'] == 'Manager': base += 10000
    elif row['Job_Role'] == 'Analyst': base += 5000
    if row['Industry'] == 'Finance': base += 5000
    elif row['Industry'] == 'Healthcare': base += 3000
    return base + np.random.randint(-1500, 1500)

df['Salary'] = df.apply(generate_salary, axis=1)

for col in ['Education', 'Job_Role', 'Industry']:
    df[col] = LabelEncoder().fit_transform(df[col])

print("🔹 Head of Data:\n", df.head())
print("\n🔹 Tail of Data:\n", df.tail())

X = df.drop('Salary', axis=1)
y = df['Salary']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"\n📊 R² Score: {round(r2*100, 2)}%")
print(f"📉 MAE: ₹{round(mae, 2)}")
print(f"📉 MSE: ₹{round(mse, 2)}")

plt.figure(figsize=(7, 5))
plt.scatter(y_test, y_pred, color='blue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.title('Actual vs Predicted Salaries')
plt.xlabel('Actual Salary')
plt.ylabel('Predicted Salary')
plt.grid(True)
plt.tight_layout()
plt.show()

importances = model.feature_importances_
features = X.columns
plt.figure(figsize=(6, 4))
sns.barplot(x=importances, y=features, palette='viridis')
plt.title("Feature Importance")
plt.tight_layout()
plt.show()

residuals = y_test - y_pred
plt.figure(figsize=(7, 4))
plt.plot(residuals.values, marker='o', linestyle='-', color='red')
plt.title('Residual Errors')
plt.xlabel('Test Sample Index')
plt.ylabel('Error (Actual - Predicted)')
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 4))
sns.kdeplot(y_test, label="Actual", shade=True, color='green')
sns.kdeplot(y_pred, label="Predicted", shade=True, color='purple')
plt.title('Distribution of Actual vs Predicted Salaries')
plt.xlabel('Salary')
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 5))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Feature Correlation")
plt.tight_layout()
plt.show()
