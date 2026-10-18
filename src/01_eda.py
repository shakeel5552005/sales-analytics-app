import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 5)

os.makedirs("reports/charts", exist_ok=True)

def save_chart(name):
    path = f"reports/charts/{name}.png"
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close()
    print(f"  saved: {path}")

# ─── 1. READ DIRECTLY FROM ORIGINAL FILE ───
print("Reading data...")
df = pd.read_csv("data/sales_data.csv")
print("Original Columns:", df.columns.tolist())

# ─── 2. COLUMNS KO ASSIGNMENT KE MUTABIQ MAP KAREIN ───
rename_map = {
    "Date": "Order_Date",
    "Order_Quantity": "Quantity",
    "Revenue": "Total_Sale"
}
df = df.rename(columns=rename_map)

# Missing columns ko default values se banayein
if "Order_ID" not in df.columns: df["Order_ID"] = range(1, len(df) + 1)
if "Customer_ID" not in df.columns: df["Customer_ID"] = range(1000, 1000 + len(df))
if "Customer_Name" not in df.columns: df["Customer_Name"] = "Customer_" + df["Customer_ID"].astype(str)
if "Brand" not in df.columns: df["Brand"] = df["Product"].str.split().str[0]
if "Region" not in df.columns: df["Region"] = df["State"]
if "City" not in df.columns: df["City"] = df["State"]
if "Discount" not in df.columns: df["Discount"] = 0.0
if "Shipping_Cost" not in df.columns: df["Shipping_Cost"] = 0.0
if "Payment_Method" not in df.columns: df["Payment_Method"] = "Credit Card"
if "Delivery_Days" not in df.columns: df["Delivery_Days"] = 3
if "Customer_Rating" not in df.columns: df["Customer_Rating"] = 4.0
if "Returned" not in df.columns: df["Returned"] = 0
if "Salesperson" not in df.columns: df["Salesperson"] = "Sales_Rep_" + (df.index % 5).astype(str)

# ─── 3. CLEANING & CONVERTING DATE ───
df["Order_Date"] = pd.to_datetime(df["Order_Date"], errors="coerce")
df = df.dropna(subset=["Profit", "Total_Sale"])

# ─── 4. FEATURE ENGINEERING ───
df["Order_Year"]  = df["Order_Date"].dt.year
df["Order_Month"] = df["Order_Date"].dt.month
df["Month_Name"]  = df["Order_Date"].dt.strftime("%b")
df["Quarter"]     = df["Order_Date"].dt.quarter
df["Day_Name"]    = df["Order_Date"].dt.day_name()

df["Profit_Margin"] = np.where(df["Total_Sale"] != 0, (df["Profit"] / df["Total_Sale"]) * 100, 0)
df["Delivery_Status"] = "Fast"
df["Order_Size_Category"] = pd.cut(df["Quantity"], bins=[0, 2, 5, 100], labels=["Small", "Medium", "Large"])

# Save Cleaned Data (App isi ko use karegi)
df.to_csv("data/sales_clean.csv", index=False)
print("✅ Cleaned data saved to data/sales_clean.csv")

# ─── 5. EDA CHARTS ───
print("\n--- Generating Charts ---")

# Monthly Sales
monthly = df.groupby(df["Order_Date"].dt.to_period("M")).agg(Sales=("Total_Sale","sum"), Profit=("Profit","sum")).reset_index()
monthly["Order_Date"] = monthly["Order_Date"].astype(str)

plt.figure()
sns.lineplot(data=monthly, x="Order_Date", y="Sales", marker="o")
plt.title("Monthly Sales"); plt.xticks(rotation=45); save_chart("01_monthly_sales")

# Sales by Category
plt.figure()
df.groupby("Product_Category")["Total_Sale"].sum().sort_values().plot(kind="barh")
plt.title("Sales by Category"); save_chart("03_sales_category")

# Profit by Category
plt.figure()
df.groupby("Product_Category")["Profit"].sum().sort_values().plot(kind="barh", color="teal")
plt.title("Profit by Category"); save_chart("04_profit_category")

# Correlation Heatmap
plt.figure(figsize=(10, 6))
num_cols = ["Quantity","Unit_Price","Profit","Total_Sale","Profit_Margin"]
sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap"); save_chart("14_heatmap")

print("\n✅ All Done! Now run 02_model.py")