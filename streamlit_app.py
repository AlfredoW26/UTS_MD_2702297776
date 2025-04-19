import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load('random_forest_model.pkl')
onehot_encoder = joblib.load('onehot_encoders.pkl')

def input_to_df(input_data):
    columns = ['no_of_adults', 'no_of_children', 'no_of_weekend_nights', 'no_of_week_nights',
               'type_of_meal_plan','required_car_parking_space','room_type_reserved','lead_time','arrival_year',
               'arrival_month','arrival_date','market_segment_type','repeated_guest','no_of_previous_cancellations',
               'no_of_previous_bookings_not_canceled','avg_price_per_room','no_of_special_requests']
    return pd.DataFrame([input_data], columns=columns)

def predict_with_model(model, user_input):
    prediction = model.predict(user_input)
    return prediction[0]

def main():
    st.title('Model Deployment UTS')
    st.info('This app will predict booking status is cancelled or not!')

    with st.expander('*Data*'):
        df = pd.read_csv('Dataset_B_hotel.csv')
        st.write(df)
        
    no_of_adults = st.number_input('Number of Adults', min_value=0, max_value=10)
    no_of_children = st.number_input('Number of Children', min_value=0,  max_value=10)
    no_of_weekend_nights = st.number_input('Number of Weekend Nights', min_value=0, max_value=8)
    no_of_week_nights = st.number_input('Number of Week Nights', min_value=0, max_value=20)
    type_of_meal_plan = st.selectbox('Meal Plan', ['Meal Plan 1', 'Meal Plan 2', 'Meal Plan 3', 'Not Selected'])
    required_car_parking_space = st.selectbox('Required Car Parking Space', [0, 1])
    room_type_reserved = st.selectbox('Room Type Reserved', ['Room Type 1', 'Room Type 2', 'Room Type 3', 'Room Type 4', 'Room Type 5', 'Room Type 6', 'Room Type 7'])
    lead_time = st.number_input('Lead Time', min_value=0, max_value=1000)
    arrival_year = st.selectbox('Arrival Year', [2017, 2018])
    arrival_month = st.number_input('Arrival Month', min_value=1, max_value=12, value=3)
    arrival_date = st.number_input('Arrival Date', min_value=1, max_value=31, value=28)
    market_segment_type = st.selectbox('Market Segment Type', ['Aviation', 'Complementary', 'Corporate', 'Offline', 'Online'])
    repeated_guest = st.selectbox('Repeated Guest', [0, 1])
    no_of_previous_cancellations = st.number_input('Number of Previous Cancellations', min_value=0, max_value=20)
    no_of_previous_bookings_not_canceled = st.number_input('Number of Previous Bookings Not Canceled', min_value=0, max_value=100)
    avg_price_per_room = st.number_input('Average Price per Room', min_value=0.0, max_value=100000.00)
    no_of_special_requests = st.number_input('Number of Special Requests', min_value=0, value=5)
    
    user_input = [no_of_adults, no_of_children, no_of_weekend_nights, no_of_week_nights, type_of_meal_plan, required_car_parking_space, room_type_reserved,
                  lead_time, arrival_year, arrival_month, arrival_date, market_segment_type, repeated_guest, no_of_previous_cancellations, no_of_previous_bookings_not_canceled,
                  avg_price_per_room, no_of_special_requests]
    
    df_input = input_to_df(user_input)
    st.write("📊 Data Input oleh User")
    st.write(df_input)

    if st.button("Prediction"):
        encode_arrival_year = 1 if arrival_year == "2018" else 0

        onehot_columns = ['type_of_meal_plan','room_type_reserved','market_segment_type']
        onehot_df = pd.DataFrame(onehot_encoder.fit_transform(df_input[onehot_columns]),
                                 columns=onehot_encoder.get_feature_names_out(onehot_columns),
                                 index=df_input.index
        )
        
        df_encoded = pd.concat([df_input.drop(columns=onehot_columns), onehot_df],axis=1)
        prediction = predict_with_model(model, df_encoded)

        st.success(f"Prediction: {'Not Cancelled' if prediction == 0 else 'Cancelled'}")

if __name__ == "__main__":
    main()
