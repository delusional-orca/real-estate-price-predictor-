import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

np.random.seed(42)
n_samples = 500
sqft = np.random.normal(2000, 500, n_samples).round()
bedrooms = np.random.choice([2, 3, 4, 5], size=n_samples, p=[0.2, 0.5, 0.2, 0.1])
age = np.random.uniform(0, 50, n_samples).round()
true_price = 50000 + (150 * sqft) + (25000 * bedrooms) - (1000 * age)
noise = np.random.normal(0, 30000, n_samples)
observed_price = true_price + noise

df = pd.DataFrame({'SqFt': sqft, 'Bedrooms': bedrooms, 'Age': age, 'Price': observed_price})

X = df[['SqFt', 'Bedrooms', 'Age']]
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

print("=" * 50)
print("       REAL ESTATE PRICE PREDICTOR ENGINE       ")
print("=" * 50)

y_pred = model.predict(X_test)
residuals = y_test - y_pred

try:
    user_sqft = float(input("Enter House Size (in SqFt, e.g., 2500): "))
    user_beds = int(input("Enter Number of Bedrooms (e.g., 3): "))
    if user_beds < 2 or user_beds > 5:
        raise ValueError("Bedrooms must be between 2 and 5.")
    user_age = float(input("Enter Age of the House (in years, e.g., 10): "))

    user_features = pd.DataFrame([[user_sqft, user_beds, user_age]], columns=['SqFt', 'Bedrooms', 'Age'])
    predicted_price = model.predict(user_features)[0]

    rse = np.sqrt(np.sum(residuals**2) / (len(residuals) - 2))
    lower_bound = predicted_price - (1.96 * rse)
    upper_bound = predicted_price + (1.96 * rse)

    print("-" * 50)
    print(f"Base Predicted Market Value: ${predicted_price:,.2f}")
    print(f"95% Confidence Range (Accounting for Noise): ${lower_bound:,.2f} to ${upper_bound:,.2f}")
    print("-" * 50)
except ValueError as e:
    print(f"\n[Error] {e}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.5, ax=ax1, color='teal')
ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Perfect Fit')
ax1.set_title(f"Model Accuracy (R² = {r2_score(y_test, y_pred):.3f})")
ax1.set_xlabel("Actual Price")
ax1.set_ylabel("Predicted Price")

sns.histplot(residuals, kde=True, ax=ax2, color='coral')
ax2.set_title("Gaussian Noise Model Verification")
ax2.set_xlabel("Residual Error Amount")

plt.tight_layout()
plt.show()