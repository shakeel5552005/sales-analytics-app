import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import glob

st.set_page_config(page_title="Sales Analytics & Prediction", layout="wide")

# ==========================================
# 1. DIRECT FILE PATHS
# ==========================================
CSV_PATH = 'src/sales_data.csv' 
MODEL_PATHS = ['src/model.pkl', 'src/random_forest.pkl', 'src/rf_model.pkl', 'model.pkl'] 

@st.cache_data
def load_data():
    if os.path.exists(CSV_PATH):
        return pd.read_csv(CSV_PATH)
    files = glob.glob('**/*.csv', recursive=True)
    if files:
        return pd.read_csv(files[0])
    return None

@st.cache_resource
def load_model():
    for path in MODEL_PATHS:
        if os.path.exists(path):
            return joblib.load(path)
    files = glob.glob('**/*.pkl', recursive=True) + glob.glob('**/*.joblib', recursive=True)
    if files:
        return joblib.load(files[0])
    return None

df = load_data()
model = load_model()

# ==========================================
# 2. HEADER & METRICS
# ==========================================
st.title("📊 Sales Performance Analytics & Profit Prediction")
st.write("Welcome to the interactive dashboard!")

if df is None:
    st.error(f"⚠️ Dataset nahi mila! Check karein ke file '{CSV_PATH}' par mojood hai.")
    st.stop()

st.subheader("Dataset Preview")
st.dataframe(df.head(10))

st.subheader("Key Metrics")
col1, col2, col3 = st.columns(3)

total_revenue = df['Total_Sale'].sum() if 'Total_Sale' in df.columns else 0
total_profit = df['Profit'].sum() if 'Profit' in df.columns else 0
total_orders = df['Order_ID'].nunique() if 'Order_ID' in df.columns else len(df)

with col1: st.metric("Total Revenue", f"${total_revenue:,.2f}")
with col2: st.metric("Total Profit", f"${total_profit:,.2f}")
with col3: st.metric("Total Orders", f"{total_orders:,}")

st.markdown("---")

# ==========================================
# 3. PREDICTION SECTION
# ==========================================
st.subheader("🔮 Predict Profit")

if model is None:
    st.warning("⚠️ Model file nahi mili! Pehle 02_model.py run karein.")
else:
    st.write("Enter the main details below to predict the profit:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        quantity = st.number_input("Quantity", min_value=1, value=10, step=1)
        unit_price = st.number_input("Unit Price ($)", min_value=1.0, value=100.0, step=1.0)
        
    with col2:
        countries = df['Country'].dropna().astype(str).unique().tolist() if 'Country' in df.columns else ["Canada", "Australia", "USA"]
        country = st.selectbox("Country", countries)
        
        categories = df['Product_Category'].dropna().astype(str).unique().tolist() if 'Product_Category' in df.columns else ["Accessories", "Bikes"]
        category = st.selectbox("Product Category", categories)
        
    with col3:
        sub_categories = df['Sub_Category'].dropna().astype(str).unique().tolist() if 'Sub_Category' in df.columns else ["Bike Racks", "Helmets"]
        sub_category = st.selectbox("Sub Category", sub_categories)
        
        genders = df['Customer_Gender'].dropna().astype(str).unique().tolist() if 'Customer_Gender' in df.columns else ["M", "F"]
        gender = st.selectbox("Customer Gender", genders)

    if st.button("Calculate Predicted Profit", type="primary"):
        try:
            # User Inputs
            input_dict = {
                'Quantity': int(quantity),
                'Unit_Price': float(unit_price),
                'Country': str(country),
                'Product_Category': str(category),
                'Sub_Category': str(sub_category),
                'Customer_Gender': str(gender)
            }
            
            # Model ki required columns
            required_columns = ['Brand', 'Day', 'Shipping_Cost', 'Year', 'State', 'Product', 'City', 
                                'Order_Year', 'Salesperson', 'Order_Size_Category', 'Delivery_Status', 
                                'Age_Group', 'Delivery_Days', 'Customer_Age', 'Month', 'Payment_Method', 
                                'Order_Month', 'Discount', 'Quarter', 'Region']
            
            # Missing columns ko dataset se fill karein (NaN Error Fix)
            for col in required_columns:
                if col in df.columns:
                    if df[col].dtype == 'object' or df[col].dtype == 'string' or df[col].dtype == 'category': 
                        # String data: Mode nikaalein, agar empty ho toh "Unknown"
                        mode_val = df[col].mode()
                        input_dict[col] = str(mode_val[0]) if not mode_val.empty else "Unknown"
                    else:
                        # Numeric data: Mean nikaalein, agar NaN ho toh 0
                        mean_val = df[col].mean()
                        input_dict[col] = float(mean_val) if not pd.isna(mean_val) else 0.0
                else:
                    input_dict[col] = 0.0 
            
            # DataFrame banayein
            input_data = pd.DataFrame([input_dict])
            
            # **NaN Error Fix**: Ensure karein ke koi bhi NaN value na bache
            input_data = input_data.fillna(0)
            
            # Model ke features ke mutabiq reorder karein
            if hasattr(model, 'feature_names_in_'):
                input_data = input_data[model.feature_names_in_]
            
            # Sab columns ko numeric/string mein convert karein jesa model expect kar raha hai
            for col in input_data.columns:
                if input_data[col].dtype == 'object':
                    input_data[col] = input_data[col].astype(str)

            # Prediction
            prediction = model.predict(input_data)
            
            st.success(f"✅ Estimated Profit: **${prediction[0]:,.2f}**")
            st.info("Note: Jo details aapne nahi di (Brand, City, Month etc.), unke liye dataset ka average/mode use kiya gaya hai.")
            
        except Exception as e:
            st.error(f"Prediction Error: {e}")