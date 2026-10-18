import pandas as pd
import numpy as np
import os

# ==========================================
# 1. DATA LOAD
# ==========================================
print("Loading data...")
df = pd.read_csv('data/sales_data.csv')
print(f"Original shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# ==========================================
# 2. MISSING VALUES CHECK
# ==========================================
print("\n--- Missing values before cleaning ---")
print(df.isnull().sum()[df.isnull().sum() > 0])

# ==========================================
# 3. CLEANING: Har column ka sahi type set karein
# ==========================================

# Date column ko datetime banayein (agar hai)
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Order_Date'] = df['Date']
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['Month_Name'] = df['Date'].dt.month_name()
    df['Day_Name'] = df['Date'].dt.day_name()
    df['Quarter'] = df['Date'].dt.quarter
    df['Order_Year'] = df['Year']
    df['Order_Month'] = df['Month']

# Numeric columns: NaN ko 0 se fill
numeric_cols = ['Customer_Age', 'Order_Quantity', 'Unit_Cost', 'Unit_Price', 'Profit', 'Cost', 'Revenue']
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

# Text columns: NaN ko "Unknown" ya mode se fill
text_cols = ['Day', 'Month', 'Age_Group', 'Customer_Gender', 'Country', 'State',
             'Product_Category', 'Sub_Category', 'Product', 'Month_Name', 'Day_Name']
for col in text_cols:
    if col in df.columns:
        # Agar column mostly numbers hain (jaise Day, Month), toh numeric banao
        if col in ['Day', 'Month']:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
        else:
            # Mode (sabse common value) se fill
            mode_val = df[col].mode()
            fill_val = mode_val[0] if not mode_val.empty else "Unknown"
            df[col] = df[col].fillna(fill_val)

# ==========================================
# 4. EXTRA COLUMNS JO 02_model.py EXPECT KARTA HAI
# ==========================================
# Ye columns aapke CSV mein nahi hain, lekin model ne training mein dekhe the.
# Hum inhe CSV ki values se derive karenge taake encoder crash na ho.

# Brand (Product se)
if 'Brand' not in df.columns and 'Product' in df.columns:
    df['Brand'] = df['Product'].apply(lambda x: str(x).split()[0] if pd.notna(x) else "Unknown")

# Region (Country se)
if 'Region' not in df.columns and 'Country' in df.columns:
    region_map = {
        'Canada': 'North America',
        'United States': 'North America',
        'Australia': 'Oceania',
        'United Kingdom': 'Europe',
        'France': 'Europe',
        'Germany': 'Europe',
    }
    df['Region'] = df['Country'].map(region_map).fillna("Unknown")

# City (State se)
if 'City' not in df.columns and 'State' in df.columns:
    df['City'] = df['State'].fillna("Unknown")

# Discount (0 se fill)
if 'Discount' not in df.columns:
    df['Discount'] = 0.0

# Shipping_Cost (0 se fill)
if 'Shipping_Cost' not in df.columns:
    df['Shipping_Cost'] = 0.0

# Payment_Method (default)
if 'Payment_Method' not in df.columns:
    df['Payment_Method'] = "Credit Card"

# Delivery_Days (0 se fill)
if 'Delivery_Days' not in df.columns:
    df['Delivery_Days'] = 0

# Salesperson (default)
if 'Salesperson' not in df.columns:
    df['Salesperson'] = "Unknown"

# Delivery_Status (default)
if 'Delivery_Status' not in df.columns:
    df['Delivery_Status'] = "Delivered"

# Order_Size_Category (Quantity se)
if 'Order_Size_Category' not in df.columns and 'Order_Quantity' in df.columns:
    df['Order_Size_Category'] = df['Order_Quantity'].apply(
        lambda x: 'Large' if x > 15 else ('Medium' if x > 5 else 'Small')
    )

# Customer_Rating (default)
if 'Customer_Rating' not in df.columns:
    df['Customer_Rating'] = 3.0

# Returned (default)
if 'Returned' not in df.columns:
    df['Returned'] = 0

# Profit_Margin (calculate from Profit/Revenue)
if 'Profit_Margin' not in df.columns and 'Profit' in df.columns and 'Revenue' in df.columns:
    df['Profit_Margin'] = df.apply(
        lambda row: (row['Profit'] / row['Revenue'] * 100) if row['Revenue'] != 0 else 0, axis=1
    )

# ==========================================
# 5. FINAL SAFETY: Koi bhi NaN bacha ho toh bhar do
# ==========================================
for col in df.columns:
    if df[col].isna().any():
        if df[col].dtype == 'object':
            df[col] = df[col].fillna("Unknown")
        else:
            df[col] = df[col].fillna(0)

# ==========================================
# 6. SAVE CLEAN DATA
# ==========================================
os.makedirs('data', exist_ok=True)
output_path = 'data/sales_clean.csv'
df.to_csv(output_path, index=False)

print(f"\n✅ Cleaned data saved to: {output_path}")
print(f"Final shape: {df.shape}")
print(f"Final columns: {list(df.columns)}")
print(f"\nMissing values after cleaning:")
print(df.isnull().sum().sum(), "total missing values")