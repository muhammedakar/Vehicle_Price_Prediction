import streamlit as st
from streamlit import components
import pandas as pd
import numpy as np
import plotly.express as px
import joblib

model = joblib.load("deployment/final_model.pkl")

st.set_page_config(
    page_title="Vehicle Price Prediction",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)

st.title('Vehicle Price Prediction')

brand = ['Maruti', 'Skoda', 'Honda', 'Hyundai', 'Toyota', 'Ford', 'Renault',
         'Mahindra', 'Tata', 'Chevrolet', 'Datsun', 'Jeep', 'Mercedes-Benz',
         'Mitsubishi', 'Audi', 'Volkswagen', 'BMW', 'Nissan', 'Lexus',
         'Jaguar', 'Land', 'MG', 'Volvo', 'Daewoo', 'Kia', 'Fiat', 'Force',
         'Ambassador', 'Ashok', 'Isuzu', 'Opel']

fuels = ['Diesel', 'Petrol', 'Other']
seller_type = ['Individual', 'Dealer', 'Trustmark Dealer']
transmission_type = ['Manual', 'Automatic']
owner_type = ['First Owner', 'Second Owner', 'Other']
seats_count = ['5', '7', 'Other']

car_brand = st.selectbox('Brand of Vehicle', options=brand)

c1, c2, c3 = st.columns(3)
co1, co2 = st.columns(2)

with co1:
    fuel = st.selectbox('Fuel Type of Vehicle', options=fuels)
with co2:
    seat = st.selectbox('Seat Count', options=seats_count)
with c1:
    seller = st.selectbox('Seller Type', options=seller_type)
with c2:
    owner = st.selectbox('Owner Type', options=owner_type)
with c3:
    trans = st.selectbox('Transmission Type', options=transmission_type)

col1, col2, col3 = st.columns(3)
cols1, cols2, cols3 = st.columns(3)

with col1:
    year = st.number_input('Model Year', value=1990, step=1, min_value=1990, max_value=2024)
with col2:
    RPM = st.number_input('RPM', value=300, step=1, min_value=300, max_value=30000)
with col3:
    TORQUE = st.number_input('TORQUE', value=10, step=1, min_value=1,max_value=1000)
with cols1:
    MILEAGE = st.number_input('MILEAGE', value=5, step=1, min_value=5,max_value=40)
with cols2:
    ENGINE = st.number_input('ENGINE', value=600, step=1, min_value=600, max_value=5000)
with cols3:
    MAX_POWER = st.number_input('MAX POWER', value=30, step=1, min_value=30, max_value= 400)

KM = st.slider("Vehicle Driven KM", 0, 300000, 25)

if car_brand == 'Ashok':
    car = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
elif car_brand == 'Audi':
    car = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
elif car_brand == 'BMW':
    car = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
elif car_brand == 'Chevrolet':
    car = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
elif car_brand == 'Daewoo':
    car = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
elif car_brand == 'Datsun':
    car = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
elif car_brand == 'Fiat':
    car = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
elif car_brand == 'Force':
    car = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
elif car_brand == 'Ford':
    car = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
elif car_brand == 'Honda':
    car = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
elif car_brand == 'Hyundai':
    car = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
elif car_brand == 'Isuzu':
    car = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
elif car_brand == 'Jaguar':
    car = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
elif car_brand == 'Jeep':
    car = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
elif car_brand == 'Kia':
    car = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
elif car_brand == 'Land':
    car = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
elif car_brand == 'Lexus':
    car = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
elif car_brand == 'MG':
    car = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
elif car_brand == 'Mahindra':
    car = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
elif car_brand == 'Maruti':
    car = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
elif car_brand == 'Mercedes-Benz':
    car = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
elif car_brand == 'Mitsubishi':
    car = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
elif car_brand == 'Nissan':
    car = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
elif car_brand == 'Opel':
    car = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
elif car_brand == 'Renault':
    car = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
elif car_brand == 'Skoda':
    car = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
elif car_brand == 'Tata':
    car = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
elif car_brand == 'Toyota':
    car = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
elif car_brand == 'Volkswagen':
    car = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
elif car_brand == 'Volvo':
    car = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
elif car_brand == 'Ambassador':
    car = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
else:
    st.title('There is a Error!')

if fuel == 'Petrol':
    fuel_a = [1, 0]
elif fuel == 'Other':
    fuel_a = [0, 1]
elif fuel == 'Diesel':
    fuel_a = [0, 0]
else:
    st.title('There is a Error!')

if seller == 'Individual':
    seller_a = [1, 0]
elif seller == 'Trustmark Dealer':
    seller_a = [0, 1]
elif seller == 'Dealer':
    seller_a = [0, 0]
else:
    st.title('There is a Error!')

if trans == 'Manual':
    trans_a = [1]
elif trans == 'Automatic':
    trans_a = [0]
else:
    st.title('There is a Error!')

if owner == 'Other':
    owner_a = [1, 0]
elif owner == 'Second Owner':
    owner_a = [0, 1]
elif owner == 'First Owner':
    owner_a = [0, 0]
else:
    st.title('There is a Error!')

if seat == '7':
    seat_a = [1, 0]
elif seat == 'Other':
    seat_a = [0, 1]
elif seat == '5':
    seat_a = [0, 0]
else:
    st.title('There is a Error!')


def predict_review_score(year, km_driven, RPM, TORQUE, MILEAGE, ENGINE, MAX_POWER, ENGINE_POWER_RATIO, FUEL_EFF_POWER,
                         Power_per_Liter,
                         Fuel_Efficiency_to_Power, Power_per_RPM, fuel_Petrol, fuel_Rare, seller_type_Individual,
                         seller_type_Trustmark_Dealer,
                         transmission_Manual, owner_Rare, owner_Second_Owner, seats_7, seats_Rare, BRAND_Ashok,
                         BRAND_Audi, BRAND_BMW, BRAND_Chevrolet,
                         BRAND_Daewoo, BRAND_Datsun, BRAND_Fiat, BRAND_Force, BRAND_Ford, BRAND_Honda, BRAND_Hyundai,
                         BRAND_Isuzu, BRAND_Jaguar, BRAND_Jeep,
                         BRAND_Kia, BRAND_Land, BRAND_Lexus, BRAND_MG, BRAND_Mahindra, BRAND_Maruti,
                         BRAND_Mercedes_Benz, BRAND_Mitsubishi, BRAND_Nissan,
                         BRAND_Opel, BRAND_Renault, BRAND_Skoda, BRAND_Tata, BRAND_Toyota, BRAND_Volkswagen,
                         BRAND_Volvo):
    features = [year, km_driven, RPM, TORQUE, MILEAGE, ENGINE, MAX_POWER, ENGINE_POWER_RATIO, FUEL_EFF_POWER,
                Power_per_Liter,
                Fuel_Efficiency_to_Power, Power_per_RPM, fuel_Petrol, fuel_Rare, seller_type_Individual,
                seller_type_Trustmark_Dealer,
                transmission_Manual, owner_Rare, owner_Second_Owner, seats_7, seats_Rare, BRAND_Ashok, BRAND_Audi,
                BRAND_BMW, BRAND_Chevrolet,
                BRAND_Daewoo, BRAND_Datsun, BRAND_Fiat, BRAND_Force, BRAND_Ford, BRAND_Honda, BRAND_Hyundai,
                BRAND_Isuzu, BRAND_Jaguar, BRAND_Jeep,
                BRAND_Kia, BRAND_Land, BRAND_Lexus, BRAND_MG, BRAND_Mahindra, BRAND_Maruti, BRAND_Mercedes_Benz,
                BRAND_Mitsubishi, BRAND_Nissan,
                BRAND_Opel, BRAND_Renault, BRAND_Skoda, BRAND_Tata, BRAND_Toyota, BRAND_Volkswagen, BRAND_Volvo]

    prediction = model.predict([features])

    return prediction


if st.button('Predict'):
    predicted_score = predict_review_score(year, KM, RPM, TORQUE,
                                           MILEAGE, ENGINE, MAX_POWER, ENGINE/MAX_POWER,
                                           MILEAGE/MAX_POWER, MAX_POWER/ENGINE, MILEAGE/MAX_POWER,
                                           MAX_POWER/RPM, *fuel_a, *seller_a, *trans_a, *owner_a,
                                           *seat_a,*car)
    st.success(f"🤩 {predicted_score[0]:.2f} €")
