import os
# Suppress TensorFlow logging warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

class LSTMForecaster:
    """
    LSTM Neural Network for forecasting the hourly energy demand
    of the school campus.
    """
    def __init__(self, lookback=24):
        self.lookback = lookback
        self.scaler_x = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.model = None
        self.feature_cols = [
            'school_load', 'ambient_temp', 'solar_irradiance', 
            'rainfall_rate', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos'
        ]
        
    def _create_features(self, df):
        """Creates cyclical and temporal features for the LSTM."""
        df_feats = df.copy()
        df_feats['hour_sin'] = np.sin(2 * np.pi * df_feats['hour'] / 24.0)
        df_feats['hour_cos'] = np.cos(2 * np.pi * df_feats['hour'] / 24.0)
        df_feats['day_sin'] = np.sin(2 * np.pi * df_feats['day_of_week'] / 7.0)
        df_feats['day_cos'] = np.cos(2 * np.pi * df_feats['day_of_week'] / 7.0)
        return df_feats
        
    def prepare_data(self, df, is_training=True):
        """
        Processes the dataframe and creates (X, y) sliding window datasets.
        
        Parameters:
        - df: Raw hourly dataframe
        - is_training: If True, fits the MinMaxScaler. Otherwise, uses the fitted one.
        
        Returns:
        - X: Numpy array of shape (samples, lookback, num_features)
        - y: Numpy array of shape (samples, 1) representing the load at next hour
        - indices: The timestamps matching each sample in y
        """
        df_feats = self._create_features(df)
        data_x = df_feats[self.feature_cols].values
        data_y = df_feats[['school_load']].values
        
        if is_training:
            scaled_x = self.scaler_x.fit_transform(data_x)
            scaled_y = self.scaler_y.fit_transform(data_y)
        else:
            scaled_x = self.scaler_x.transform(data_x)
            scaled_y = self.scaler_y.transform(data_y)
            
        X, y = [], []
        # We start extracting after the lookback window
        for i in range(self.lookback, len(df)):
            X.append(scaled_x[i - self.lookback:i])
            y.append(scaled_y[i])
            
        X = np.array(X)
        y = np.array(y)
        indices = df.index[self.lookback:]
        
        return X, y, indices

    def build_model(self, input_shape):
        """Constructs the LSTM network architecture."""
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.1),
            LSTM(30, return_sequences=False),
            Dropout(0.1),
            Dense(20, activation='relu'),
            Dense(1)  # Predicts continuous load (scaled)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        self.model = model
        return model
        
    def train(self, X_train, y_train, validation_split=0.1, epochs=8, batch_size=64):
        """Trains the LSTM model."""
        if self.model is None:
            input_shape = (X_train.shape[1], X_train.shape[2])
            self.build_model(input_shape)
            
        print(f"Training LSTM model for {epochs} epochs...")
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1
        )
        return history
        
    def predict(self, X):
        """Predicts energy demand and inverse-scales the result back to kW."""
        scaled_preds = self.model.predict(X, verbose=0)
        unscaled_preds = self.scaler_y.inverse_transform(scaled_preds)
        return unscaled_preds.flatten()

    def train_and_forecast_all(self, df, train_split=0.8, epochs=8):
        """
        Orchestrates full train/test split, training, and runs predictions 
        across the entire year. For the initial lookback period, it falls 
        back to raw load data.
        
        Returns:
        - df_result: Dataframe with an added 'predicted_load' column.
        - metrics: Dict containing train/test MAE and RMSE.
        """
        # Prepare sliding windows
        X, y, indices = self.prepare_data(df, is_training=True)
        
        # Split into train and test indices
        split_idx = int(len(X) * train_split)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Train
        self.train(X_train, y_train, epochs=epochs)
        
        # Predict on entire dataset
        all_preds = self.predict(X)
        
        # Build results DataFrame
        df_result = df.copy()
        # For the first 'lookback' hours, we can't forecast using LSTM, so we use actual load
        df_result['predicted_load'] = df_result['school_load']
        # Fill in the rest with LSTM predictions
        df_result.loc[indices, 'predicted_load'] = all_preds
        
        # Calculate evaluation metrics
        actual_train = self.scaler_y.inverse_transform(y_train).flatten()
        preds_train = self.predict(X_train)
        
        actual_test = self.scaler_y.inverse_transform(y_test).flatten()
        preds_test = self.predict(X_test)
        
        train_mae = np.mean(np.abs(actual_train - preds_train))
        test_mae = np.mean(np.abs(actual_test - preds_test))
        
        train_rmse = np.sqrt(np.mean((actual_train - preds_train)**2))
        test_rmse = np.sqrt(np.mean((actual_test - preds_test)**2))
        
        metrics = {
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse
        }
        
        return df_result, metrics

if __name__ == "__main__":
    # Test file script execution
    from data_generator import generate_chennai_weather_and_load
    print("Testing LSTM Forecaster...")
    data = generate_chennai_weather_and_load()
    
    forecaster = LSTMForecaster()
    df_out, metrics = forecaster.train_and_forecast_all(data, train_split=0.8, epochs=2)
    print("\nTraining complete!")
    print(f"Train MAE: {metrics['train_mae']:.2f} kW")
    print(f"Test MAE: {metrics['test_mae']:.2f} kW")
    print(df_out[['school_load', 'predicted_load']].head(30))
