import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

class SensorSimulator:
    def generate_sensor_data(self, days=365):
        np.random.seed(42)
        timestamps = pd.date_range(start='2024-01-01', periods=days, freq='D')
        
        data = {
            'date': timestamps,
            'temperature': 20 + 10 * np.sin(np.arange(days) * 2 * np.pi / 365) + np.random.normal(0, 2, days),
            'humidity': 60 + 20 * np.sin(np.arange(days) * 2 * np.pi / 365) + np.random.normal(0, 5, days),
            'soil_moisture': 40 + 15 * np.cos(np.arange(days) * 2 * np.pi / 365) + np.random.normal(0, 3, days),
            'ph_level': 6.5 + np.random.normal(0, 0.3, days),
            'nitrogen': 50 + np.random.normal(0, 5, days),
            'phosphorus': 30 + np.random.normal(0, 3, days),
            'potassium': 40 + np.random.normal(0, 4, days),
            'rainfall': np.maximum(0, np.random.exponential(5, days)),
            'solar_radiation': 200 + 100 * np.sin(np.arange(days) * 2 * np.pi / 365) + np.random.normal(0, 20, days)
        }
        
        df = pd.DataFrame(data)
        
        df['crop_yield'] = (
            0.3 * df['temperature'] +
            0.2 * df['humidity'] +
            0.25 * df['soil_moisture'] +
            0.1 * df['nitrogen'] +
            0.05 * df['phosphorus'] +
            0.05 * df['potassium'] +
            0.05 * df['rainfall'] +
            np.random.normal(0, 5, days)
        )
        
        return df

class CropYieldPredictor:
    def __init__(self, sequence_length=7):
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler()
        self.model = None
    
    def prepare_sequences(self, data):
        scaled_data = self.scaler.fit_transform(data)
        X, y = [], []
        
        for i in range(len(scaled_data) - self.sequence_length):
            X.append(scaled_data[i:i+self.sequence_length])
            y.append(scaled_data[i+self.sequence_length, -1])
        
        return np.array(X), np.array(y)
    
    def build_model(self, input_shape):
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
    
    def train(self, df):
        features = df.drop(['date', 'crop_yield'], axis=1).values
        targets = df[['crop_yield']].values
        data = np.column_stack([features, targets])
        
        X, y = self.prepare_sequences(data)
        
        train_size = int(0.8 * len(X))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        self.model = self.build_model((X_train.shape[1], X_train.shape[2]))
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=50,
            batch_size=32,
            verbose=0
        )
        
        test_loss, test_mae = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"Test MAE: {test_mae:.4f}")
        
        return history, X_test, y_test
    
    def predict(self, recent_data):
        scaled_data = self.scaler.transform(recent_data)
        prediction = self.model.predict(scaled_data.reshape(1, self.sequence_length, -1), verbose=0)
        
        dummy = np.zeros((1, scaled_data.shape[1]))
        dummy[0, -1] = prediction[0, 0]
        yield_prediction = self.scaler.inverse_transform(dummy)[0, -1]
        
        return yield_prediction

class IrrigationController:
    def recommend_irrigation(self, soil_moisture, temperature, humidity):
        if soil_moisture < 30:
            water_amount = 50
            urgency = "High"
        elif soil_moisture < 45:
            water_amount = 30
            urgency = "Medium"
        else:
            water_amount = 0
            urgency = "Low"
        
        if temperature > 30 and humidity < 40:
            water_amount += 20
        
        return {
            'water_amount_mm': water_amount,
            'urgency': urgency,
            'recommendation': f"Apply {water_amount}mm of water" if water_amount > 0 else "No irrigation needed"
        }

def main():
    print("=== Smart Agriculture AI-IoT System ===\n")
    
    print("1. Generating sensor data...")
    simulator = SensorSimulator()
    df = simulator.generate_sensor_data(days=365)
    df.to_csv('sensor_data.csv', index=False)
    print(f"Generated {len(df)} days of sensor data\n")
    
    print("2. Training crop yield prediction model...")
    predictor = CropYieldPredictor(sequence_length=7)
    history, X_test, y_test = predictor.train(df)
    print("Model training complete\n")
    
    print("3. Testing prediction on recent data...")
    recent_data = df.drop(['date', 'crop_yield'], axis=1).iloc[-7:].values
    predicted_yield = predictor.predict(recent_data)
    print(f"Predicted crop yield: {predicted_yield:.2f} kg/hectare\n")
    
    print("4. Irrigation recommendation...")
    controller = IrrigationController()
    current_moisture = df['soil_moisture'].iloc[-1]
    current_temp = df['temperature'].iloc[-1]
    current_humidity = df['humidity'].iloc[-1]
    
    irrigation = controller.recommend_irrigation(current_moisture, current_temp, current_humidity)
    print(f"Soil Moisture: {current_moisture:.1f}%")
    print(f"Temperature: {current_temp:.1f}°C")
    print(f"Recommendation: {irrigation['recommendation']}")
    print(f"Urgency: {irrigation['urgency']}")

if __name__ == '__main__':
    main()