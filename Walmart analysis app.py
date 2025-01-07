import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Load Dataset
@st.cache_data
def load_data():
    data = pd.read_csv("Walmart_Sales.csv")
    data['Date'] = pd.to_datetime(data['Date'], format='%d-%m-%Y')
    return data

data = load_data()

# Custom CSS
st.markdown("""
    <style>
        h1, h2, h3, h4 {
            color: #003366;
        }
        .stApp {
            background-color: #f9f9f9;
        }
        .sidebar .sidebar-content {
            background-color: #e6e6e6;
        }
        hr {
            border: 1px solid #003366;
        }
        .metric-label {
            color: #003366;
        }
        footer {
            font-size: small;
            color: gray;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Navigation")
pages = [
    "Dataset Overview",
    "Missing Value Analysis",
    "Correlation Heatmap",
    "Pair Plot",
    "Weekly Sales Trend",
    "Holiday Impact Analysis",
    "Temperature vs. Sales",
    "Outlier Detection",
    "Sales Distribution",
    "Sales by Store",
    "Predictive Analysis"
]
selected_page = st.sidebar.radio("Go to", pages)

# Calendar Input
selected_date = st.sidebar.date_input("Select a Date", value=pd.to_datetime("2010-02-05"))

# Function to Add Footer
def add_footer():
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<footer>Developed by Areesha Rizwan</footer>", unsafe_allow_html=True)

# Page 1: Dataset Overview
if selected_page == "Dataset Overview":
    st.title("Dataset Overview")
    st.write("### This dataset includes sales data from Walmart stores, exploring the impact of holidays, temperature, fuel prices, and economic factors.")
    st.write(data.head())
    st.write(data.describe())
    add_footer()

# Page 2: Missing Value Analysis
elif selected_page == "Missing Value Analysis":
    st.title("Missing Value Analysis")
    missing_values = data.isnull().sum()
    st.write(missing_values)
    st.bar_chart(missing_values)
    add_footer()

# Page 3: Correlation Heatmap
elif selected_page == "Correlation Heatmap":
    st.title("Correlation Heatmap")
    numeric_data = data.select_dtypes(include=['number'])
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)
    add_footer()

# Page 4: Pair Plot
elif selected_page == "Pair Plot":
    st.title("Pair Plot")
    pair_data = data[["Weekly_Sales", "Temperature", "Fuel_Price", "CPI", "Unemployment"]]
    pair_fig = sns.pairplot(pair_data, diag_kind="kde", plot_kws={"alpha": 0.6})
    st.pyplot(pair_fig)
    add_footer()

# Page 5: Weekly Sales Trend
elif selected_page == "Weekly Sales Trend":
    st.title("Weekly Sales Trend")
    sales_trend = px.line(data, x="Date", y="Weekly_Sales", title="Weekly Sales Over Time", 
                          color_discrete_sequence=["#003366"])
    st.plotly_chart(sales_trend)
    add_footer()

# Page 6: Holiday Impact Analysis
elif selected_page == "Holiday Impact Analysis":
    st.title("Holiday Impact on Sales")
    avg_sales = data.groupby("Holiday_Flag")["Weekly_Sales"].mean().reset_index()
    avg_sales["Holiday_Flag"] = avg_sales["Holiday_Flag"].map({0: "Non-Holiday", 1: "Holiday"})
    holiday_bar = px.bar(avg_sales, x="Holiday_Flag", y="Weekly_Sales", color="Holiday_Flag", 
                         title="Average Weekly Sales by Holiday Status",
                         color_discrete_sequence=["#003366", "#66b3ff"])
    st.plotly_chart(holiday_bar)
    add_footer()

# Page 7: Temperature vs. Sales
elif selected_page == "Temperature vs. Sales":
    st.title("Temperature vs. Weekly Sales")
    scatter_fig = px.scatter(data, x="Temperature", y="Weekly_Sales", trendline="ols",
                             title="Impact of Temperature on Weekly Sales",
                             color_discrete_sequence=["#003366"])
    st.plotly_chart(scatter_fig)
    add_footer()

# Page 8: Outlier Detection
elif selected_page == "Outlier Detection":
    st.title("Outlier Detection")
    box_fig = px.box(data, x="Store", y="Weekly_Sales", title="Weekly Sales Distribution by Store",
                     color_discrete_sequence=["#66b3ff"])
    st.plotly_chart(box_fig)
    add_footer()

# Page 9: Sales Distribution
elif selected_page == "Sales Distribution":
    st.title("Sales Distribution")
    fig, ax = plt.subplots()
    sns.histplot(data["Weekly_Sales"], bins=20, kde=True, ax=ax)
    st.pyplot(fig)
    add_footer()

# Page 10: Sales by Store
elif selected_page == "Sales by Store":
    st.title("Total Sales by Store")
    store_sales = data.groupby("Store")["Weekly_Sales"].sum().reset_index()
    store_bar = px.bar(store_sales, x="Store", y="Weekly_Sales", title="Total Sales by Store",
                       color_discrete_sequence=["#003366"])
    st.plotly_chart(store_bar)
    add_footer()

# Page 11: Predictive Analysis
elif selected_page == "Predictive Analysis":
    st.title("Predictive Analysis")
    X = data[["Temperature", "Fuel_Price", "CPI", "Unemployment"]]
    y = data["Weekly_Sales"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestRegressor(random_state=42)
    model.fit(X_scaled, y)

    temperature = st.slider("Select Temperature", min_value=int(data["Temperature"].min()), 
                             max_value=int(data["Temperature"].max()), value=70)
    fuel_price = st.slider("Select Fuel Price", min_value=float(data["Fuel_Price"].min()), 
                            max_value=float(data["Fuel_Price"].max()), value=3.0)
    cpi = st.slider("Select CPI", min_value=float(data["CPI"].min()), max_value=float(data["CPI"].max()), value=200.0)
    unemployment = st.slider("Select Unemployment Rate", min_value=float(data["Unemployment"].min()), 
                              max_value=float(data["Unemployment"].max()), value=5.0)

    features = np.array([[temperature, fuel_price, cpi, unemployment]])
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)

    st.write(f"### Predicted Weekly Sales: **${prediction[0]:,.2f}**")
    add_footer()
