import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import glob

st.set_page_config(page_title="Sales Analytics & Prediction", layout="wide")

# ==========================================
# 1. FILE PATHS (Apni file ka naam yahan check karein)
# ==========================================
CSV_FILE_NAME = 'src/sales_data.csv' # <--- Agar aapki file ka naam kuch aur hai toh yahan badal lein

# ==========================================
# 2. LOAD DATA & MODEL
# ==========================================
@st.cache_data
def load_data():
    if os.path.exists(CSV_FILE_NAME):
        return pd.read_csv(CSV_FILE_NAME)
    else:
        files = glob.glob('**/*.csv', recursive=True)
        if files:
            return pd.read_csv(files[0])
    return None

@st.cache_resource
def load_model():
    files = glob.glob('**/*.pkl', recursive=True) + glob.glob('**/*.joblib', recursive=True)
    if files:
        return joblib.load(files[0])
    return None

df = load_data()
model = load_model()

# ==========================================
# 3. HEADER & METRICS
# ==========================================
st.title("📊 Sales Performance Analytics & Profit Prediction")

if df is None:
    st.error("⚠️ Dataset nahi mila! Check karein ke CSV file mojood hai.")
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
# 4. PREDICTION SECTION
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
        countries = df['Country'].unique().tolist() if 'Country' in df.columns else ["Canada", "Australia", "USA"]
        country = st.selectbox("Country", countries)
        
        categories = df['Product_Category'].unique().tolist() if 'Product_Category' in df.columns else ["Accessories", "Bikes"]
        category = st.selectbox("Product Category", categories)
        
    with col3:
        sub_categories = df['Sub_Category'].unique().tolist() if 'Sub_Category' in df.columns else ["Bike Racks", "Helmets"]
        sub_category = st.selectbox("Sub Category", sub_categories)
        
        genders = df['Customer_Gender'].unique().tolist() if 'Customer_Gender' in df.columns else ["M", "F"]
        gender = st.selectbox("Customer Gender", genders)

    if st.button("Calculate Predicted Profit", type="primary"):
        try:
            # Ek dictionary banayein jisme sirf wahi columns hon jo model maang raha hai
            # Hum baaki columns ke liye dataset ka "Mode" (sab se common value) ya "Mean" (average) uthayenge
            
            input_dict = {
                'Quantity': quantity,
                'Unit_Price': unit_price,
                'Country': country,
                'Product_Category': category,
                'Sub_Category': sub_category,
                'Customer_Gender': gender
            }
            
            # Baaki missing columns ke liye default values (Mode/Mean) dataset se nikalenge
            missing_cols = ['Brand', 'Day', 'Shipping_Cost', 'Year', 'State', 'Product', 'City', 
                            'Order_Year', 'Salesperson', 'Order_Size_Category', 'Delivery_Status', 
                            'Age_Group', 'Delivery_Days', 'Customer_Age', 'Month', 'Payment_Method', 
                            'Order_Month', 'Discount', 'Quarter', 'Region']
            
            for col in missing_cols:
                if col in df.columns:
                    if df[col].dtype == 'object' or df[col].dtype == 'str':
                        # Agar text column hai toh sab se common value (Mode) lein
                        input_dict[col] = df[col].mode()[0] 
                    else:
                        # Agar number column hai toh Average lein
                        input_dict[col] = df[col].mean()
            
            # DataFrame banayein (sirf ek row ka)
            input_data = pd.DataFrame([input_dict])
            
            # Prediction
            prediction = model.predict(input_data)
            
            st.success(f"✅ Estimated Profit: **${prediction[0]:,.2f}**")
            st.info("Note: Baaki details (Brand, City, Month, etc.) ke liye model ne dataset ka Average/Mode use kiya hai.")
            
        except Exception as e:
            st.error(f"Prediction Error: {e}")