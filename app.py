import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import glob

st.set_page_config(page_title="Sales Analytics & Prediction", layout="wide")

# ==========================================
# 1. SMART FILE FINDER
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
    csv_path = find_file('.csv')
    if csv_path:
        return pd.read_csv(csv_path)
    return None

@st.cache_resource
def load_model():
    model_path = find_file('.pkl')
    if not model_path: model_path = find_file('.joblib')
    if model_path:
        return joblib.load(model_path)
    return None

df = load_data()
model = load_model()

# ==========================================
# 2. HEADER
# ==========================================
st.title("📊 Sales Performance Analytics & Profit Prediction")
st.write("Welcome to the interactive dashboard!")

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
# 3. PREDICTION SECTION
# ==========================================
st.subheader("🔮 Predict Profit")

if model is None:
    st.warning("⚠️ Model file nahi mili!")
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
            # 1. User ke diye hue inputs
            input_dict = {
                'Quantity': quantity,
                'Unit_Price': unit_price,
                'Country': country,
                'Product_Category': category,
                'Sub_Category': sub_category,
                'Customer_Gender': gender
            }
            
            # 2. Model ko chahiye ye saare columns (Jo error mein aaye the)
            required_columns = ['Brand', 'Day', 'Shipping_Cost', 'Year', 'State', 'Product', 'City', 
                                'Order_Year', 'Salesperson', 'Order_Size_Category', 'Delivery_Status', 
                                'Age_Group', 'Delivery_Days', 'Customer_Age', 'Month', 'Payment_Method', 
                                'Order_Month', 'Discount', 'Quarter', 'Region']
            
            # 3. Har missing column ke liye dataset se Mode ya Mean uthana
            for col in required_columns:
                if col in df.columns:
                    # Agar text (object) hai toh sab se common value
                    if df[col].dtype == 'object' or df[col].dtype == 'string': 
                        input_dict[col] = df[col].mode()[0] if not df[col].mode().empty else "Unknown"
                    else:
                        # Agar number hai toh average value
                        input_dict[col] = df[col].mean()
                else:
                    # Agar column dataset mein hi nahi hai (Safety ke liye)
                    input_dict[col] = 0 
            
            # 4. DataFrame banayein aur predict karein
            input_data = pd.DataFrame([input_dict])
            
            # **Zaroori Step:** Model ki feature order ke mutabiq columns sort karein
            if hasattr(model, 'feature_names_in_'):
                input_data = input_data[model.feature_names_in_]
            
            prediction = model.predict(input_data)
            
            st.success(f"✅ Estimated Profit: **${prediction[0]:,.2f}**")
            st.info("Note: Baaki details (Brand, City, Month, etc.) ke liye dataset ka average/mode use kiya gaya hai.")
            
        except Exception as e:
            st.error(f"Prediction Error: {e}")