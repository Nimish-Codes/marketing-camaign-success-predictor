import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Load dataset
data = pd.read_csv("KAG_conversion_data.csv")

data.dropna(inplace=True)

# Data preprocessing
# Calculate success percent based on converted customers and ad spent
data['success_percent'] = (data['Approved_Conversion'] / data['Total_Conversion']) * 100

# Convert categorical variables to one-hot encoded representation
data = pd.get_dummies(data, columns=['gender', 'age'])

# Handle missing values in the target variable
y = data['success_percent']
y.fillna(y.mean(), inplace=True)

# Convert column names
data.rename(columns={'impression': 'Impressions', 'clicks': 'Clicks', 'spent': 'Spent'}, inplace=True)

# Split data into features and target variable
X = data.drop(columns=['success_percent', 'Approved_Conversion', 'Total_Conversion'])  # Features

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Model evaluation
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)

# Define function to predict success percent
def predict_success_percent(num_ads, Impressions, age_group, gender, Clicks, Spent):
    input_data = {
        'ad_id': [None],  # Placeholder for ad_id (not used in prediction)
        'xyz_campaign_id': [None],  # Placeholder for xyz_campaign_id (not used in prediction)
        'fb_campaign_id': [None],  # Placeholder for fb_campaign_id (not used in prediction)
        'Impressions': [Impressions],
        'Clicks': [Clicks],
        'Spent': [Spent],
    }
    # Encode gender and age group
    for col in X.columns:
        if col.startswith('gender_' + gender):
            input_data[col] = [1]
        elif col.startswith('age_' + age_group):
            input_data[col] = [1]
        else:
            input_data[col] = [0]
    input_df = pd.DataFrame(input_data)

    # Ensure columns order matches the order during training
    input_df = input_df[X.columns]

    prediction = model.predict(input_df)
    return prediction

# Streamlit UI
st.title('Ad Success Prediction')

# User inputs
num_ads = st.slider('Number of Ads', min_value=1, max_value=10, value=5)
Impressions = st.number_input('Impressions', value=10000)
age_group = st.selectbox('Age Group', options=['30-34', '35-39', '40-44', '45-49'])
gender = st.radio('Gender', options=['M', 'F'])
Clicks = st.number_input('Clicks', value=200)
Spent = st.number_input('Spent', value=1000)

# Prediction
if st.button('Predict Success Percent'):
    success_percent = predict_success_percent(num_ads, Impressions, age_group, gender, Clicks, Spent)
    st.write(f"Predicted Success Percent: {success_percent}")
