import xgboost as xgb
import streamlit as st
import pandas as pd

#Loading up the Regression model we created
model = xgb.XGBRegressor()
model.load_model('xgb_model.json')

#Caching the model for faster loading
@st.cache


# Define the prediction function
def predict(Lanes, Visability, Competition , Brand, Suds , table, x, y, z):
    #Predicting the price of the Lanes
    if Visability  == 'Fair':
        Visability = 0
    elif Visability == 'Good':
        Visability = 1
    elif Visability == 'Very Good':
        Visability = 2
    elif Visability == 'Premium':
        Visability = 3
    elif Visability == 'Ideal':
        Visability = 4

    if Competition  == 'J':
        Competition  = 0
    elif Competition  == 'I':
        Competition  = 1
    elif Competition  == 'H':
        Competition  = 2
    elif Competition  == 'G':
        Competition  = 3
    elif Competition  == 'F':
        Competition  = 4
    elif Competition  == 'E':
        Competition  = 5
    elif Competition  == 'D':
        Competition  = 6

    if Brand == 'I1':
        Brand = 0
    elif Brand == 'SI2':
        Brand = 1
    elif Brand == 'SI1':
        Brand = 2
    elif Brand == 'VS2':
        Brand = 3
    elif Brand == 'VS1':
        Brand = 4
    elif Brand == 'VVS2':
        Brand = 5
    elif Brand == 'VVS1':
        Brand = 6
    elif Brand == 'IF':
        Brand = 7


    prediction = model.predict(pd.DataFrame([[Lanes, Visability, Competition , Brand, Suds , table, x, y, z]], columns=['Lanes', 'Visability', 'Competition ', 'Brand', 'Suds ', 'table', 'x', 'y', 'z']))
    return prediction


st.title('Go To Market Predictor')
st.image("""https://internationalautospa.com/wp-content/uploads/before-and-after-1.png""")
st.header('Enter the characteristics of the site:')
Lanes = st.number_input('Lanes Weight:', min_value=0.1, max_value=10.0, value=1.0)
Visability = st.selectbox('Visability Rating:', ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'])
Competition  = st.selectbox('Competition  Rating:', ['J', 'I', 'H', 'G', 'F', 'E', 'D'])
Brand = st.selectbox('Brand Rating:', ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF'])
Suds  = st.number_input('Site Suds  Percentage:', min_value=0.1, max_value=100.0, value=1.0)
table = st.number_input('Site Table Percentage:', min_value=0.1, max_value=100.0, value=1.0)
x = st.number_input('Site Length (X) in mm:', min_value=0.1, max_value=100.0, value=1.0)
y = st.number_input('Site Width (Y) in mm:', min_value=0.1, max_value=100.0, value=1.0)
z = st.number_input('Site Height (Z) in mm:', min_value=0.1, max_value=100.0, value=1.0)

if st.button('Predict Price'):
    price = predict(Lanes, Visability, Competition , Brand, Suds , table, x, y, z)
    st.success(f'The predicted price/Member Count (or whatever) of the Site is ${price[0]:.2f} USD')
