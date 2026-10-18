import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import glob

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(page_title="Sales Analytics & Profit Prediction", layout="wide")

# ==========================================
# 1. SMART FILE FINDER (CSV & MODEL)
# ==========================================
def find_file(extension, folder="."):
    search_paths = [f"{folder}/*{extension}", f"{folder}/src/*{extension}"]
    for path in search_paths:
        files = glob.glob(path)
        if files:
            return files[0]
    return None

@st.cache_data
def load_data():
    csv_path = find_file('src/sales_data.csv')
    if csv_path:
        return pd.read_csv(csv_path)
    return None

@st.cache_resource
def load_model():
    model_path = find_file('.pkl')
    if not model_path:
        model_path = find_file('.joblib')
    if model_path:
        return joblib.load(model_path)
    return None

df = load_data()
model = load_model()

# ==========================================
# 2. HEADER & METRICS
# ==========================================
st.title("📊 Sales Performance Analytics & Profit Prediction")
st.write("Welcome to the interactive dashboard!")

if df is None:
    st.error("⚠️ Dataset (.csv file) nahi mili! Please ensure koi CSV file folder mein mojood hai.")
    st.stop()

st.subheader("Dataset Preview")
st.dataframe(df.head(10))

st.subheader("Key Metrics")
col1, col2, col3 = st.columns(3)

# Aapke dataset ke mutabiq yahan column names diye hain (Aapke CSV ke hisaab se)
total_revenue = df['Revenue'].sum() if 'Revenue' in df.columns else 0
total_profit = df['Profit'].sum() if 'Profit' in df.columns else 0
total_orders = len(df)

with col1:
    st.metric("Total Revenue", f"${total_revenue:,.2f}")
with col2:
    st.metric("Total Profit", f"${total_profit:,.2f}")
with col3:
    st.metric("Total Orders", f"{total_orders:,}")

st.markdown("---")

# ==========================================
# 3. PREDICTION SECTION (SMART AUTO-FILL)
# ==========================================
st.subheader("🔮 Predict Profit")

if model is None:
    st.warning("⚠️ Model file nahi mili! Please pehle `02_model.py` run karein.")
else:
    st.write("Enter the main details below to predict the profit:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        quantity = st.number_input("Quantity", min_value=1, value=10, step=1)
        # Aapke dataset mein 'Unit_Cost' aur 'Unit_Price' dono hain
        unit_cost = st.number_input("Unit Cost ($)", min_value=1.0, value=45.0, step=1.0)
        unit_price = st.number_input("Unit Price ($)", min_value=1.0, value=120.0, step=1.0)
        
    with col2:
        countries = df['Country'].unique().tolist() if 'Country' in df.columns else ["Canada", "Australia", "USA"]
        country = st.selectbox("Country", countries)
        
        categories = df['Product_Category'].unique().tolist() if 'Product_Category' in df.columns else ["Accessories", "Bikes"]
        category = st.selectbox("Product Category", categories)
        
        sub_categories = df['Sub_Category'].unique().tolist() if 'Sub_Category' in df.columns else ["Bike Racks", "Helmets"]
        sub_category = st.selectbox("Sub Category", sub_categories)
        
    with col3:
        genders = df['Customer_Gender'].unique().tolist() if 'Customer_Gender' in df.columns else ["M", "F"]
        gender = st.selectbox("Customer Gender", genders)
        
        # Agar aapke dataset mein state hai
        states = df['State'].unique().tolist() if 'State' in df.columns else ["British Columbia", "California"]
        state = st.selectbox("State", states)

    if st.button("Calculate Predicted Profit", type="primary"):
        try:
            # 1. Ek base row banayein aapke CSV ki pehli row se (jo saare columns hogi)
            input_data = df.head(1).copy()
            
            # 2. User ke diye hue inputs is row mein update karein
            input_data['Quantity'] = quantity
            if 'Unit_Cost' in input_data.columns: input_data['Unit_Cost'] = unit_cost
            if 'Unit_Price' in input_data.columns: input_data['Unit_Price'] = unit_price
            if 'Country' in input_data.columns: input_data['Country'] = country
            if 'Product_Category' in input_data.columns: input_data['Product_Category'] = category
            if 'Sub_Category' in input_data.columns: input_data['Sub_Category'] = sub_category
            if 'Customer_Gender' in input_data.columns: input_data['Customer_Gender'] = gender
            if 'State' in input_data.columns: input_data['State'] = state
            
            # 3. Jo columns target hain (Profit, Cost, Revenue) unko drop kar dein
            # Kyunki model inko predict karta hai, input nahi leta
            cols_to_drop = ['Profit', 'Cost', 'Revenue', 'Date', 'Order_ID', 'Customer_ID', 'Customer_Name']
            input_data = input_data.drop(columns=[c for c in cols_to_drop if c in input_data.columns], errors='ignore')

            # 4. Model ke expected features ke mutabiq columns order karein
            if hasattr(model, 'feature_names_in_'):
                input_data = input_data[model.feature_names_in_]
            
            # 5. Har column ko safely string ya numeric banayein (NaN error se bachne ke liye)
            for col in input_data.columns:
                if input_data[col].dtype == 'object':
                    input_data[col] = input_data[col].fillna("Unknown").astype(str)
                else:
                    input_data[col] = pd.to_numeric(input_data[col], errors='coerce').fillna(0)

            # 6. Prediction
            prediction = model.predict(input_data)
            
            st.success(f"✅ Estimated Profit: **${prediction[0]:,.2f}**")
            st.info("Note: Jo details aapne nahi di (Brand, City, Month, etc.), unke liye dataset ki values use hui hain.")
            
        except Exception as e:
            st.error(f"Prediction Error: {e}")