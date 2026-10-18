import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import glob

st.set_page_config(page_title="Sales Analytics & Profit Prediction", layout="wide")

# ==========================================
# 1. FILE FINDER
# ==========================================
CSV_FILE_NAME = 'src/sales_data.csv'  # Aapki CSV ka path

def find_model_file():
    # Poore project folder mein recursive search (.pkl / .joblib)
    patterns = ['**/*.pkl', '**/*.joblib', '**/*.sav']
    for pattern in patterns:
        files = glob.glob(pattern, recursive=True)
        if files:
            return files[0]
    return None

@st.cache_data
def load_data():
    if os.path.exists(CSV_FILE_NAME):
        return pd.read_csv(CSV_FILE_NAME)
    files = glob.glob('**/*.csv', recursive=True)
    if files:
        return pd.read_csv(files[0])
    return None

@st.cache_resource
def load_model():
    path = find_model_file()
    if path:
        return joblib.load(path)
    return None

df = load_data()
model = load_model()

# ==========================================
# 2. DEBUG INFO (temporary)
# ==========================================
with st.expander("🔧 Debug Info (click to check file paths)"):
    st.write("Current working directory:", os.getcwd())
    st.write("CSV Path:", CSV_FILE_NAME, "| Exists:", os.path.exists(CSV_FILE_NAME))
    st.write("Model Path found:", find_model_file())
    st.write("Files in current folder:", os.listdir('.'))

# ==========================================
# 3. HEADER
# ==========================================
st.title("📊 Sales Performance Analytics & Profit Prediction")
st.write("Welcome to the interactive dashboard!")

if df is None:
    st.error("⚠️ Dataset (.csv) nahi mili!")
    st.stop()

st.subheader("Dataset Preview")
st.dataframe(df.head(10))

st.subheader("Key Metrics")
col1, col2, col3 = st.columns(3)

total_revenue = df['Revenue'].sum() if 'Revenue' in df.columns else 0
total_profit = df['Profit'].sum() if 'Profit' in df.columns else 0
total_orders = len(df)

with col1: st.metric("Total Revenue", f"${total_revenue:,.2f}")
with col2: st.metric("Total Profit", f"${total_profit:,.2f}")
with col3: st.metric("Total Orders", f"{total_orders:,}")

st.markdown("---")

# ==========================================
# 4. PREDICTION
# ==========================================
st.subheader("🔮 Predict Profit")

if model is None:
    st.warning("⚠️ Model file nahi mili! Pehle 02_model.py run karein.")
    st.info("Jab 02_model.py chalayein, dhyaan dein ke wo model kis naam aur kis folder mein save ho raha hai. Phir us file ko project ke root folder mein copy kar dein.")
else:
    st.success(f"✅ Model loaded from: {find_model_file()}")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        quantity = st.number_input("Quantity", min_value=1, value=10, step=1)
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
        
        states = df['State'].unique().tolist() if 'State' in df.columns else ["British Columbia", "California"]
        state = st.selectbox("State", states)

    if st.button("Calculate Predicted Profit", type="primary"):
        try:
            input_data = df.head(1).copy()
            input_data['Quantity'] = quantity
            if 'Unit_Cost' in input_data.columns: input_data['Unit_Cost'] = unit_cost
            if 'Unit_Price' in input_data.columns: input_data['Unit_Price'] = unit_price
            if 'Country' in input_data.columns: input_data['Country'] = country
            if 'Product_Category' in input_data.columns: input_data['Product_Category'] = category
            if 'Sub_Category' in input_data.columns: input_data['Sub_Category'] = sub_category
            if 'Customer_Gender' in input_data.columns: input_data['Customer_Gender'] = gender
            if 'State' in input_data.columns: input_data['State'] = state
            
            cols_to_drop = ['Profit', 'Cost', 'Revenue', 'Date', 'Order_ID', 'Customer_ID', 'Customer_Name']
            input_data = input_data.drop(columns=[c for c in cols_to_drop if c in input_data.columns], errors='ignore')

            if hasattr(model, 'feature_names_in_'):
                input_data = input_data[model.feature_names_in_]
            
            for col in input_data.columns:
                if input_data[col].dtype == 'object':
                    input_data[col] = input_data[col].fillna("Unknown").astype(str)
                else:
                    input_data[col] = pd.to_numeric(input_data[col], errors='coerce').fillna(0)

            prediction = model.predict(input_data)
            st.success(f"✅ Estimated Profit: **${prediction[0]:,.2f}**")
            
        except Exception as e:
            st.error(f"Prediction Error: {e}")