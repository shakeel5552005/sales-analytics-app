import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import glob
import traceback
import plotly.express as px
import plotly.graph_objects as go

# ==========================================
# 1. PAGE CONFIG & CUSTOM CSS (PROFESSIONAL DESIGN)
# ==========================================
st.set_page_config(
    page_title="Sales Analytics Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional look
st.markdown("""
    <style>
    /* Main background */
    .stApp {
        background-color: #f8f9fa;
    }
    
    /* Header styling */
    h1, h2, h3 {
        color: #1f2937;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Custom metric cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #ffffff 0%, #f1f5f9 100%);
        border: 1px solid #e2e8f0;
        padding: 15px 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s;
    }
    
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }
    
    /* Metric label */
    div[data-testid="stMetricLabel"] {
        font-size: 14px;
        color: #64748b;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Metric value */
    div[data-testid="stMetricValue"] {
        font-size: 26px;
        color: #0f172a;
        font-weight: 700;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
        color: white;
    }
    
    section[data-testid="stSidebar"] * {
        color: #f1f5f9 !important;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #ffffff;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        border: 1px solid #e2e8f0;
        border-bottom: none;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #3b82f6 !important;
        color: white !important;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        border: none;
        width: 100%;
        transition: all 0.2s;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        transform: scale(1.02);
    }
    
    /* Hide Streamlit footer */
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. FILE PATHS & LOADERS
# ==========================================
def find_csv():
    for p in ['data/sales_clean.csv', 'src/sales_clean.csv', 'src/sales_data.csv', 'sales_data.csv']:
        if os.path.exists(p):
            return p
    files = glob.glob('**/*.csv', recursive=True)
    return files[0] if files else None

def find_model():
    for p in ['models/best_model.pkl', 'best_model.pkl']:
        if os.path.exists(p):
            return p
    files = glob.glob('**/*.pkl', recursive=True)
    return files[0] if files else None

CSV_PATH = find_csv()
MODEL_PATH = find_model()

@st.cache_data
def load_data():
    if CSV_PATH and os.path.exists(CSV_PATH):
        return pd.read_csv(CSV_PATH)
    return None

@st.cache_resource
def load_model():
    if MODEL_PATH and os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

df = load_data()
model = load_model()

# ==========================================
# 3. SIDEBAR (Navigation + Info)
# ==========================================
with st.sidebar:
    st.markdown("<h1 style='text-align: center; color: #60a5fa;'>📊 SALES PRO</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #94a3b8;'>Analytics & Profit Prediction</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Data info
    st.markdown("### 📁 Data Status")
    if df is not None:
        st.success(f"✅ {len(df):,} rows loaded")
    else:
        st.error("❌ Dataset missing")
    
    if model is not None:
        st.success("✅ Model Loaded")
    else:
        st.error("❌ Model missing")
    
    st.markdown("---")
    st.markdown("### 👨‍💻 About")
    st.info("This dashboard provides comprehensive sales analytics with ML-based profit prediction.")

# ==========================================
# 4. MAIN DASHBOARD
# ==========================================
st.title("📊 Sales Performance Dashboard")
st.caption("Interactive analytics, insights, and AI-powered profit prediction.")

if df is None:
    st.error("⚠️ Dataset nahi mili! Please ensure `data/sales_clean.csv` or `src/sales_data.csv` exists.")
    st.stop()

# ==========================================
# 5. KEY METRICS (Custom Cards)
# ==========================================
st.markdown("### 💡 Key Performance Indicators")
col1, col2, col3, col4 = st.columns(4)

total_revenue = df['Revenue'].sum() if 'Revenue' in df.columns else 0
total_profit = df['Profit'].sum() if 'Profit' in df.columns else 0
total_orders = len(df)
avg_profit = total_profit / total_orders if total_orders > 0 else 0

with col1: st.metric("💰 Total Revenue", f"${total_revenue:,.2f}")
with col2: st.metric("📈 Total Profit", f"${total_profit:,.2f}")
with col3: st.metric("🛒 Total Orders", f"{total_orders:,}")
with col4: st.metric("📊 Avg Profit/Order", f"${avg_profit:,.2f}")

st.markdown("<br>", unsafe_allow_html=True)

# ==========================================
# 6. TABS (Professional Layout)
# ==========================================
tab1, tab2, tab3 = st.tabs(["📊 Overview", "📈 Advanced Graphs", "🔮 Predict Profit"])

# -------- TAB 1: OVERVIEW --------
with tab1:
    st.markdown("### 📅 Monthly Revenue Trend")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if 'Month' in df.columns and 'Revenue' in df.columns:
            month_map = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                        7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
            monthly_revenue = df.groupby('Month')['Revenue'].sum().reset_index()
            monthly_revenue['Month_Name'] = monthly_revenue['Month'].map(month_map)
            
            fig1 = px.bar(
                monthly_revenue, x='Month_Name', y='Revenue',
                title='Monthly Revenue',
                color='Revenue', color_continuous_scale='Blues',
                labels={'Month_Name': 'Month', 'Revenue': 'Revenue ($)'}
            )
            fig1.update_layout(showlegend=False, height=350, plot_bgcolor='white')
            st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        if 'Product_Category' in df.columns and 'Profit' in df.columns:
            cat_profit = df.groupby('Product_Category')['Profit'].sum().reset_index()
            fig2 = px.pie(
                cat_profit, names='Product_Category', values='Profit',
                title='Profit by Category', hole=0.5,
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig2.update_layout(height=350)
            st.plotly_chart(fig2, use_container_width=True)
    
    st.markdown("### 🌍 Top Countries by Revenue")
    if 'Country' in df.columns and 'Revenue' in df.columns:
        country_revenue = df.groupby('Country')['Revenue'].sum().reset_index()
        country_revenue = country_revenue.sort_values('Revenue', ascending=True).tail(10)
        
        fig3 = px.bar(
            country_revenue, x='Revenue', y='Country', orientation='h',
            title='Top 10 Countries',
            color='Revenue', color_continuous_scale='Greens',
            labels={'Revenue': 'Revenue ($)', 'Country': 'Country'}
        )
        fig3.update_layout(showlegend=False, height=400, plot_bgcolor='white')
        st.plotly_chart(fig3, use_container_width=True)

# -------- TAB 2: ADVANCED GRAPHS --------
with tab2:
    st.markdown("### 🔍 Deep Dive Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        qty_col = 'Order_Quantity' if 'Order_Quantity' in df.columns else ('Quantity' if 'Quantity' in df.columns else None)
        if qty_col and 'Profit' in df.columns:
            sample_df = df.sample(n=min(1000, len(df)), random_state=42)
            fig4 = px.scatter(
                sample_df, x=qty_col, y='Profit',
                color='Product_Category' if 'Product_Category' in df.columns else None,
                title=f'{qty_col} vs Profit (Sample)',
                opacity=0.6
            )
            fig4.update_layout(height=400, plot_bgcolor='white')
            st.plotly_chart(fig4, use_container_width=True)
    
    with col2:
        if 'Customer_Age' in df.columns:
            fig5 = px.histogram(
                df, x='Customer_Age', nbins=30,
                title='Customer Age Distribution',
                color_discrete_sequence=['#f59e0b']
            )
            fig5.update_layout(height=400, plot_bgcolor='white')
            st.plotly_chart(fig5, use_container_width=True)
    
    col3, col4 = st.columns(2)
    
    with col3:
        if 'Customer_Gender' in df.columns and 'Profit' in df.columns:
            gender_profit = df.groupby('Customer_Gender')['Profit'].sum().reset_index()
            fig6 = px.bar(
                gender_profit, x='Customer_Gender', y='Profit',
                title='Profit by Gender', color='Customer_Gender',
                color_discrete_sequence=['#10b981', '#f43f5e']
            )
            fig6.update_layout(showlegend=False, height=400, plot_bgcolor='white')
            st.plotly_chart(fig6, use_container_width=True)
    
    with col4:
        if 'Sub_Category' in df.columns and 'Revenue' in df.columns:
            sub_rev = df.groupby('Sub_Category')['Revenue'].sum().reset_index()
            sub_rev = sub_rev.sort_values('Revenue', ascending=True).tail(8)
            fig7 = px.bar(
                sub_rev, x='Revenue', y='Sub_Category', orientation='h',
                title='Top Sub-Categories',
                color='Revenue', color_continuous_scale='Purples'
            )
            fig7.update_layout(showlegend=False, height=400, plot_bgcolor='white')
            st.plotly_chart(fig7, use_container_width=True)

# -------- TAB 3: PREDICTION --------
with tab3:
    st.markdown("### 🔮 Predict Profit (AI-Powered)")
    st.write("Enter the main details below to predict the expected profit.")
    
    if model is None:
        st.warning("⚠️ Model file nahi mili! Pehle `02_model.py` run karein.")
    else:
        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                quantity = st.number_input("Quantity", min_value=1, value=10, step=1)
                unit_price = st.number_input("Unit Price ($)", min_value=1.0, value=120.0, step=1.0)
            
            with col2:
                countries = df['Country'].dropna().unique().tolist() if 'Country' in df.columns else ["Canada"]
                country = st.selectbox("Country", countries)
                
                categories = df['Product_Category'].dropna().unique().tolist() if 'Product_Category' in df.columns else ["Accessories"]
                category = st.selectbox("Product Category", categories)
                
                sub_categories = df['Sub_Category'].dropna().unique().tolist() if 'Sub_Category' in df.columns else ["Bike Racks"]
                sub_category = st.selectbox("Sub Category", sub_categories)
            
            with col3:
                genders = df['Customer_Gender'].dropna().unique().tolist() if 'Customer_Gender' in df.columns else ["M", "F"]
                gender = st.selectbox("Customer Gender", genders)
                
                states = df['State'].dropna().unique().tolist() if 'State' in df.columns else ["British Columbia"]
                state = st.selectbox("State", states)
            
            submitted = st.form_submit_button("🚀 Calculate Predicted Profit")
            
            if submitted:
                try:
                    input_dict = {}
                    
                    if hasattr(model, 'feature_names_in_'):
                        for col in model.feature_names_in_:
                            if col in ['Quantity', 'Order_Quantity']:
                                input_dict[col] = quantity
                            elif col == 'Unit_Price':
                                input_dict[col] = unit_price
                            elif col == 'Country':
                                input_dict[col] = country
                            elif col == 'Product_Category':
                                input_dict[col] = category
                            elif col == 'Sub_Category':
                                input_dict[col] = sub_category
                            elif col == 'Customer_Gender':
                                input_dict[col] = gender
                            elif col == 'State':
                                input_dict[col] = state
                            elif col in df.columns:
                                first_val = df[col].iloc[0]
                                if pd.isna(first_val):
                                    if df[col].dtype == 'object':
                                        mode_val = df[col].mode()
                                        input_dict[col] = mode_val[0] if not mode_val.empty else ""
                                    else:
                                        input_dict[col] = 0
                                else:
                                    input_dict[col] = first_val
                            else:
                                input_dict[col] = 0
                    
                    input_data = pd.DataFrame([input_dict])
                    input_data = input_data.fillna(0)
                    
                    for col in input_data.columns:
                        if input_data[col].dtype == 'object':
                            input_data[col] = input_data[col].astype(str)
                        else:
                            input_data[col] = pd.to_numeric(input_data[col], errors='coerce').fillna(0)
                    
                    prediction = model.predict(input_data)
                    st.success(f"✅ Estimated Profit: **${prediction[0]:,.2f}**")
                    st.balloons()
                    
                except Exception as e:
                    st.error(f"Prediction Error: {type(e).__name__}: {e}")