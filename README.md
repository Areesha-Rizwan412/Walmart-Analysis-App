# Walmart Sales Analysis and Prediction Project

This project is part of the **Introduction to Data Science** course, aimed at analyzing Walmart sales data, deriving insights, and building a predictive model using machine learning. The project is deployed as an interactive Streamlit web application.

## Project Overview

The project explores sales trends, holiday impacts, and the effects of economic and weather factors on weekly sales. It also provides a predictive analysis feature for estimating weekly sales based on user-input economic and weather data.

---

## Features

1. **Exploratory Data Analysis (EDA)**:
   - Missing value analysis with bar charts.
   - Correlation heatmap to visualize feature relationships.
   - Pair plots for pairwise feature analysis.
   - Weekly sales trend analysis.
   - Holiday sales comparison (holiday vs. non-holiday).
   - Temperature impact on sales.
   - Outlier detection and sales distribution analysis.
   - Aggregated total sales by store.

2. **Data Preprocessing**:
   - Handled missing values.
   - Scaled numerical features.
   - Selected relevant features for predictive modeling.

3. **Machine Learning Model**:
   - Built a `RandomForestRegressor` for weekly sales prediction.
   - Evaluated the model's performance with scaled features.

4. **Streamlit Application**:
   - Multi-page interactive web app.
   - Allows users to explore data insights.
   - Provides sliders for feature inputs to predict weekly sales.
   - Custom styling for a professional and user-friendly interface.

---

## Tools and Technologies

- **Python**: Core language used for data analysis and model development.
- **Streamlit**: For building the interactive web application.
- **Pandas, NumPy**: For data manipulation and preprocessing.
- **Seaborn, Matplotlib, Plotly**: For data visualization.
- **Scikit-learn**: For machine learning model implementation.
- **CSS**: For custom UI styling.

---

## Installation and Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Walmart-Sales-Analysis.git
   cd Walmart-Sales-Analysis

