import pandas as pd
import numpy as np
import joblib, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

os.makedirs("models", exist_ok=True)
os.makedirs("reports/charts", exist_ok=True)

# ─── 1. READ CLEANED FILE (Bina parse_dates ke) ───
print("Reading cleaned data...")
df = pd.read_csv("data/sales_clean.csv")

# Date ko alag se convert karein (agar column mojood ho)
if "Order_Date" in df.columns:
    df["Order_Date"] = pd.to_datetime(df["Order_Date"])

# ─── 2. DROP UNWANTED COLUMNS (Data Leakage) ───
drop_cols = ["Order_ID","Customer_ID","Customer_Name","Customer_Rating","Returned",
             "Total_Sale","Order_Date","Month_Name","Day_Name","Profit_Margin",
             "Profit", "Cost", "Unit_Cost"]

# Sirf woh columns drop karein jo dataset mein mojood hain
cols_to_drop = [c for c in drop_cols if c in df.columns]
X = df.drop(columns=cols_to_drop)
y = df["Profit"]

# ─── 3. PREPROCESSING ───
cat_cols = X.select_dtypes(include="object").columns.tolist()
num_cols = X.select_dtypes(include=np.number).columns.tolist()

pre = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ("num", StandardScaler(), num_cols)
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ─── 4. TRAINING MODELS ───
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree":     DecisionTreeRegressor(random_state=42),
    "Random Forest":     RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
}

results, trained = {}, {}
for name, mdl in models.items():
    print(f"Training {name}...")
    pipe = Pipeline([("pre", pre), ("model", mdl)])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    results[name] = {
        "MAE":  mean_absolute_error(y_test, preds),
        "MSE":  mean_squared_error(y_test, preds),
        "RMSE": np.sqrt(mean_squared_error(y_test, preds)),
        "R2":   r2_score(y_test, preds),
    }
    trained[name] = pipe

comparison = pd.DataFrame(results).T.sort_values("R2", ascending=False)
print("\n--- Model Comparison ---")
print(comparison)
comparison.to_csv("reports/model_comparison.csv")

# ─── 5. SAVE BEST MODEL ───
best_name = comparison.index[0]
best_model = trained[best_name]
print(f"\nBest Model: {best_name}")
joblib.dump(best_model, "models/best_model.pkl")
print("✅ Saved models/best_model.pkl")

# ─── 6. CHARTS ───
preds = best_model.predict(X_test)
plt.figure()
plt.scatter(y_test, preds, alpha=0.4)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
plt.title(f"Actual vs Predicted — {best_name}")
plt.xlabel("Actual Profit"); plt.ylabel("Predicted Profit")
plt.tight_layout(); plt.savefig("reports/charts/15_actual_vs_pred.png", dpi=100); plt.close()

print("\n✅ Modeling Complete! Now run streamlit run app.py")