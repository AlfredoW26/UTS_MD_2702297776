import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load('random_forest_model.pkl')
onehot_encoder = joblib.load('onehot_encoders.pkl')
label_encoder = joblib.load('label_encoders.pkl')

def input_to_df(input):
    df = pd.DataFrame([input], columns=[
        'no_of_adults', 'no_of_children', 'no_of_weekend_nights', 'no_of_week_nights',
        'type_of_meal_plan','required_car_parking_space','room_type_reserved','lead_time','arrival_year',
        'arrival_month','arrival_date','market_segment_type','repeated_guest','no_of_previous_cancellations',
        'no_of_previous_bookings_not_canceled','avg_price_per_room','no_of_special_requests'
    ])
    return df

def label_arrival_year(df):
    if 'arrival_year' in df.columns and 'arrival_year' in label_encoder:
        df['arrival_year'] = label_encoder['arrival_year'].transform(df['arrival_year'])
    return df

def onehot_room_type_reserved(df):
    if 'room_type_reserved' in df.columns:
        if 'room_type_reserved' in onehot_encoder:
            encoded = onehot_encoder['room_type_reserved'].transform(df[['room_type_reserved']]).toarray()
            encoded_df = pd.DataFrame(encoded, columns=onehot_encoder['room_type_reserved'].get_feature_names_out(['room_type_reserved']))
            df = df.drop('room_type_reserved', axis=1)
            df = pd.concat([df.reset_index(drop=True), encoded_df], axis=1)
        else:
            st.error("OneHot encoder untuk 'room_type_reserved' tidak ditemukan dalam file.")
    else:
        st.error("Kolom 'room_type_reserved' tidak ditemukan dalam data input.")
    return df


def onehot_type_of_meal_plan(df):
    if 'room_type_reserved' in df.columns:
        encoded = onehot_encoder['type_of_meal_plan'].transform(df[['type_of_meal_plan']]).toarray()
        encoded_df = pd.DataFrame(
            encoded,
            columns=onehot_encoder['type_of_meal_plan'].get_feature_names_out(['type_of_meal_plan']),
            index=df.index
        )
        df = pd.concat([df.drop('type_of_meal_plan', axis=1), encoded_df], axis=1)
    return df

def onehot_market_segment_type(df):
    if 'market_segment_type' in df.columns:
        encoded = onehot_encoder['market_segment_type'].transform(df[['market_segment_type']]).toarray()
        encoded_df = pd.DataFrame(
            encoded,
            columns=onehot_encoder['market_segment_type'].get_feature_names_out(['market_segment_type']),
            index=df.index
        )
        df = pd.concat([df.drop('market_segment_type', axis=1), encoded_df], axis=1)
    return df

def main():
    st.title('Model Deployment UTS')
    st.info('This app will predict booking status is cancelled or not!')

    with st.expander('**Data**'):
        df = pd.read_csv('Dataset_B_hotel.csv')
        st.write(df)
        
    no_of_adults = st.number_input('Number of Adults', min_value=0, max_value=10)
    no_of_children = st.number_input('Number of Children', min_value=0,  max_value=10)
    no_of_weekend_nights = st.number_input('Number of Weekend Nights', min_value=0, max_value=7)
    no_of_week_nights = st.number_input('Number of Week Nights', min_value=0, max_value=20)
    type_of_meal_plan = st.selectbox('Meal Plan', ['Meal Plan 1', 'Meal Plan 2', 'Meal Plan 3', 'Not Selected'])
    required_car_parking_space = st.selectbox('Required Car Parking Space', [0, 1])
    room_type_reserved = st.selectbox('Room Type Reserved', ['Room Type 1', 'Room Type 2', 'Room Type 3', 'Room Type 4', 'Room Type 5', 'Room Type 6', 'Room Type 7'])
    lead_time = st.number_input('Lead Time', min_value=0, max_value=1000)
    arrival_year = st.number_input('Arrival Year', min_value=2000, max_value=2023, value=2022)
    arrival_month = st.number_input('Arrival Month', min_value=1, max_value=12, value=3)
    arrival_date = st.number_input('Arrival Date', min_value=1, max_value=31, value=28)
    market_segment_type = st.selectbox('Market Segment Type', ['Aviation', 'Complementary', 'Corporate', 'Offline', 'Online'])
    repeated_guest = st.number_input('Repeated Guest', min_value=0, max_value=1, value=0)
    no_of_previous_cancellations = st.number_input('Number of Previous Cancellations', min_value=0, max_value=20)
    no_of_previous_bookings_not_canceled = st.number_input('Number of Previous Bookings Not Canceled', min_value=0, max_value=100)
    avg_price_per_room = st.number_input('Average Price per Room', min_value=0.0, max_value=100000.00)
    no_of_special_requests = st.number_input('Number of Special Requests', min_value=0, value=5)
    
    user_input = [no_of_adults, no_of_children, no_of_weekend_nights, no_of_week_nights, type_of_meal_plan, required_car_parking_space, room_type_reserved,
                  lead_time, arrival_year, arrival_month, arrival_date, market_segment_type, repeated_guest, no_of_previous_cancellations, no_of_previous_bookings_not_canceled,
                  avg_price_per_room, no_of_special_requests]
    
    df_input = input_to_df(user_input)
  
    st.write('Data input by user')
    st.write(df_input)
    
    df_input = label_arrival_year(df_input)
    df_input = onehot_room_type_reserved(df_input)
    # df_input = onehot_type_of_meal_plan(df_input)
    # df_input = onehot_market_segment_type(df_input)

    # # --- Encoding ---
    # df_encoded = encode(df_input)
    # df_encoded = df_encoded.reindex(columns=model.feature_names_in_, fill_value=0)

    # # --- Prediction ---
    # prediction = model.predict(df_encoded)[0]
    # prediction_proba = model.predict_proba(df_encoded)[0]
    # result = 'Booking Cancelled' if prediction == 1 else 'Booking Not Cancelled'
    # confidence = np.max(prediction_proba) * 100

    # # --- Output ---
    # st.subheader('Prediction Result')
    # st.success(f'Prediction: **{result}** with **{confidence:.2f}%** confidence')

if __name__ == "__main__":
    main()
