import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
from sklearn.preprocessing import OneHotEncoder
import pickle
import joblib

class DataHandler:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.input_df = None
        self.output_df = None
        self.label_encoders = {}
        self.onehot_encoders = {}

    def load_data(self):
        self.df = pd.read_csv(self.file_path) 

    def drop_missing_values(self):
        self.df.dropna(inplace=True)
    
    def label_encode(self, column_name):
        label_encoder = preprocessing.LabelEncoder()
        self.df[column_name] = label_encoder.fit_transform(self.df[column_name])

    def one_hot_encode(self, column_name):
        onehot_encoder = OneHotEncoder(sparse_output=False)

        encoded_array = onehot_encoder.fit_transform(self.df[column_name])
        encoded_df = pd.DataFrame(encoded_array, columns=onehot_encoder.get_feature_names_out(column_name))
        
        self.df = self.df.reset_index(drop=True)
        encoded_df = encoded_df.reset_index(drop=True)  

        self.df = pd.concat([self.df, encoded_df], axis=1)       
        self.df = self.df.drop(columns=columns_encode)

    def drop_column(self, column_name):
        self.df = self.df.drop(column_name, axis = 1)

    def create_input_output(self, target_column):
        self.output_df = self.df[target_column]
        self.input_df = self.df.drop(target_column, axis = 1)   

    def save_encoders(self, label_filename='label_encoders.pkl', onehot_filename='onehot_encoders.pkl'):
        with open(label_filename, 'wb') as f:
            pickle.dump(self.label_encoders, f)
        print(f"Label encoders saved to {label_filename}")

        with open(onehot_filename, 'wb') as f:
            pickle.dump(self.onehot_encoders, f)
        print(f"One-hot encoders saved to {onehot_filename}") 

file_path = 'Dataset_B_hotel.csv'
data_handler = DataHandler(file_path)
data_handler.load_data()
data_handler.drop_missing_values()
data_handler.label_encode('arrival_year')
data_handler.label_encode('booking_status')
columns_encode = ['type_of_meal_plan','room_type_reserved','market_segment_type']
data_handler.one_hot_encode(columns_encode)
data_handler.drop_column('Booking_ID')
data_handler.create_input_output('booking_status')
data_handler.save_encoders()

input_data = data_handler.input_df
output_data = data_handler.output_df

class RFModelHandler:
    def __init__(self, input_data, output_data):
        self.input_data = input_data
        self.output_data = output_data
        self.model = RandomForestClassifier(random_state= 42)
        self.x_train, self.x_test, self.y_train, self.y_test, self.y_predict = [None] * 5

    def split_data(self, test_size = 0.2, random_state=42):
        x = self.input_data
        y = self.output_data       
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            x, y, test_size = test_size, random_state= random_state
        )

    def train_model(self):
        self.model.fit(self.x_train, self.y_train)

    def evaluate_model(self):
        predictions = self.model.predict(self.x_test)
        accuracy = accuracy_score(self.y_test, predictions)
        print("Accuracy: ", accuracy)
        print("Classification Report:\n", classification_report(self.y_test, predictions, target_names = ['Canceled','Not_Canceled']))

    def save_model(self, filename='random_forest_model.pkl'):
        joblib.dump(self.model, filename, compress=3)

model_handler = RFModelHandler(input_data, output_data)
model_handler.split_data()
model_handler.train_model()
model_handler.evaluate_model()
model_handler.save_model()