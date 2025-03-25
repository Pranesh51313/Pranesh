import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
import pandas as pd
import numpy as np
import sqlite3
import time
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
import datetime

# Function to fetch stock data
def fetch_stock_data(tickers, period="1y", interval="1d"):
    stock_data = {}
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(period=period, interval=interval)
            if data.empty:
                st.warning(f"No data found for {ticker}. Please check the ticker symbol.")
                continue
            
            # Extract additional features
            data['Returns'] = data['Close'].pct_change()
            data['SMA_10'] = data['Close'].rolling(window=10).mean()
            data['EMA_10'] = data['Close'].ewm(span=10, adjust=False).mean()
            data['Volatility'] = data['Returns'].rolling(window=10).std()
            data.dropna(inplace=True)
            stock_data[ticker] = data
        except Exception as e:
            st.error(f"Error fetching data for {ticker}: {e}")
    return stock_data

# Function to perform exploratory data analysis
def perform_eda(data, ticker):
    st.write(f"## Preprocessing for {ticker}")
    
    # Outlier detection using boxplot
    st.write("### Outlier Detection")
    st.plotly_chart(px.box(data, y='Close', title=f'{ticker} Close Price Outliers', labels={'Close': 'Closing Price'}))
    
    # Feature names
    st.write("### Feature Names")
    st.write(list(data.columns))
    st.write(f"## EDA for {ticker}")
    
    # Price trend
    st.write("### Price Trend")
    st.plotly_chart(px.line(data, x=data.index, y='Close', title=f'{ticker} Closing Price Trend', labels={'x': 'Date', 'Close': 'Closing Price'}))
    
    # Returns distribution
    st.write("### Returns Distribution")
    st.plotly_chart(px.histogram(data, x='Returns', title=f'{ticker} Returns Distribution', labels={'Returns': 'Daily Returns'}))
    
    # Moving averages
    st.write("### Moving Averages")
    ma_fig = px.line(data, x=data.index, y=['SMA_10', 'EMA_10', 'Close'], title=f'{ticker} Moving Averages', labels={'x': 'Date', 'value': 'Stock Price'})
    st.plotly_chart(ma_fig)
    
    # Correlation heatmap
    st.write("### Correlation Matrix")
    st.plotly_chart(px.imshow(data.corr(), title=f'{ticker} Feature Correlation Heatmap', labels={'color': 'Correlation'}))

def apply_pca(X_train, X_test, n_components=2):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    explained_variance = sum(pca.explained_variance_ratio_)
    st.write(f"PCA applied. Explained variance: {explained_variance:.2f}")
    return X_train_pca, X_test_pca

# Function to train models with hyperparameter tuning
def train_models(data):
    # Accuracy before PCA
    X_raw = data[['Close', 'SMA_10', 'EMA_10', 'Volatility']].dropna()
    y_raw = data['Close'].loc[X_raw.index]
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(X_raw, y_raw, test_size=0.2, random_state=42)
    
    # Apply PCA here with the correct arguments
    X_train_pca, X_test_pca = apply_pca(X_train_raw, X_test_raw)

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=300, max_depth=20),
        "Support Vector Regression": SVR()
    }
    
    # Accuracy for each model before PCA
    st.write("### Accuracy Before PCA")
    model_accuracies = {}
    for name, model in models.items():
        model.fit(X_train_raw, y_train_raw)
        pred_raw = model.predict(X_test_raw)
        mae_raw = mean_absolute_error(y_test_raw, pred_raw)
        mse_raw = mean_squared_error(y_test_raw, pred_raw)
        r2_raw = r2_score(y_test_raw, pred_raw)
        
        model_accuracies[name] = {
            "MAE": mae_raw,
            "MSE": mse_raw,
            "R²": r2_raw
        }
        
        st.write(f"{name} - MAE: {mae_raw:.4f}, MSE: {mse_raw:.4f}, R²: {r2_raw:.4f}")
    
    # Now apply PCA for the transformed data
    X = X_train_pca
    y = y_train_raw  # Use the target variable from the raw training set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    param_grid = {
        "Random Forest": {"n_estimators": [200, 300, 400], "max_depth": [15, 20, 25]},
        "Support Vector Regression": {"C": [10, 100, 1000], "gamma": [0.01, 0.1, 1]}
    }
    
    best_model = None
    best_score = float("inf")
    best_r2 = float("-inf")
    
    # Train and evaluate models after PCA
    st.write("### Accuracy After PCA")
    for name, model in models.items():
        if name in param_grid:
            grid_search = GridSearchCV(model, param_grid[name], cv=5, scoring='r2')
            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_
        else:
            model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        if r2 > best_r2:
            best_score = mae
            best_model = name
            best_r2 = r2
        
        st.write(f"{name} - MAE: {mae:.4f}, MSE: {mse:.4f}, R²: {r2:.4f}")
    
    st.write(f"### Best Model After PCA: {best_model} with MAE: {best_score:.4f} and R²: {best_r2:.4f}")
    
    # Displaying raw accuracy results for comparison
    st.write("### Model Accuracy Comparison (Before vs After PCA)")
    comparison_df = pd.DataFrame(model_accuracies).T
    st.dataframe(comparison_df)


def apply_arima(data):
    model = ARIMA(data['Close'], order=(5,1,0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=10)
    
    # Creating date index for forecasting
    last_date = data.index[-1]
    forecast_dates = [last_date + datetime.timedelta(days=i) for i in range(1, 11)]
    
    st.write("### ARIMA Forecasting")
    fig = go.Figure()
    
    # Past data
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Actual Close Price'))
    
    # Forecast data
    fig.add_trace(go.Scatter(x=forecast_dates, y=forecast, mode='lines', name='Forecasted Close Price', line=dict(dash='dot')))
    
    fig.update_layout(title="ARIMA Forecast for Next 10 Days", xaxis_title='Date', yaxis_title='Price')
    st.plotly_chart(fig)

def apply_lstm(data):
    st.write("### LSTM Forecasting")
    
    # Scaling data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['Close']])
    
    # Prepare the data for LSTM model
    lookback = 10
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    
    # Reshape X for LSTM (samples, time steps, features)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Build the LSTM model
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Train the model
    model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=1)  # Increase epochs for better results
    
    # Make future predictions
    future_inputs = scaled_data[-lookback:]
    future_inputs = np.reshape(future_inputs, (1, lookback, 1))
    
    future_predictions = []
    for _ in range(10):
        pred = model.predict(future_inputs)
        future_predictions.append(pred[0, 0])
        future_inputs = np.roll(future_inputs, -1)
        future_inputs[0, -1, 0] = pred[0, 0]
    
    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
    
    # Creating date index for forecasting
    last_date = data.index[-1]
    forecast_dates = [last_date + datetime.timedelta(days=i) for i in range(1, 11)]
    
    # Plot the past and future predictions
    fig = go.Figure()
    
    # Past data
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Actual Close Price'))
    
    # Forecast data
    fig.add_trace(go.Scatter(x=forecast_dates, y=future_predictions.flatten(), mode='lines', name='Forecasted Close Price', line=dict(dash='dot')))
    
    fig.update_layout(title="LSTM Forecast for Next 10 Days", xaxis_title='Date', yaxis_title='Predicted Close Price')
    st.plotly_chart(fig)
 

# Streamlit App
st.title("📈 Multi-Stock Real-Time Analysis & Automated Forecasting")

tickers = st.text_input("Enter Stock Tickers (comma-separated, e.g., AAPL, MSFT, TSLA, GOOGL, AMZN)").split(',')
tickers = [t.strip().upper() for t in tickers if t.strip()]

period = st.selectbox("Select Data Period", ["1d", "5d", "1mo", "3mo", "6mo", "1y", "5y", "max"])
interval = st.selectbox("Select Data Interval", ["1m", "5m", "15m", "30m", "1h", "1d", "1wk", "1mo"])

if st.button("Apply Changes"):
    stock_data = fetch_stock_data(tickers, period, interval)
    if not stock_data:
        st.error("No valid stock data retrieved. Please check your inputs.")
    else:
        for ticker, data in stock_data.items():
            st.write(f"## {ticker} Stock Data")
            st.dataframe(data)
            
            st.write("### Summary Statistics")
            st.dataframe(data.describe())
            
            perform_eda(data, ticker)
            train_models(data)
            apply_arima(data)
            apply_lstm(data)