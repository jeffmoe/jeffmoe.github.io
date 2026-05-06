---
title: LSTM Stock Price Prediction (TensorFlow)
parent: Machine Learning and Artificial Intelligence
nav_order: 1
---

### Overview
Built an LSTM model to predict next‑day stock prices using historical Yahoo Finance data.

### Function Creation
```python
def explore_data(df, target_col='Adj Close', date_col='Date'):
    df_analysis = df.copy()
    print(f"Shape of dataset: {df_analysis.shape}")
    print(f"Missing vals in each column:\n {df_analysis.isnull().sum()}")
    print(f"Date range: {df_analysis.index.min()} to {df_analysis.index.max()}")

    print(df_analysis.head())
    print(df_analysis.info())
    print(df_analysis.describe())

    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    fig.suptitle('Stock Price Time Series Analysis', fontsize=16, y=1.02)
    axes[0] = sns.lineplot(data=df_analysis, x=df_analysis.index, y='Adj Close', ax=axes[0])
    axes[0].set_title('Adjusted Close Price Over Time')
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel('Adjusted Close Price')

    price_cols = ['High', 'Low', 'Open', 'Close']
    for col in price_cols:
        axes[1].plot(df_analysis.index, df_analysis[col], alpha=0.6, label=col)
    axes[1].set_title('All Price Features Over Time')
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('Price')
    axes[1].legend(loc='best')
    axes[1].grid(True, alpha=0.3)

    rolling_mean = df_analysis['Adj Close'].rolling(window=30).mean()
    rolling_std = df_analysis['Adj Close'].rolling(window=30).std()
    
    axes[2].plot(df_analysis.index, df_analysis['Adj Close'], label='Original', alpha=0.5, linewidth=1)
    axes[2].plot(df_analysis.index, rolling_mean, label='30-day Rolling Mean', color='red', linewidth=2)
    axes[2].fill_between(df_analysis.index, rolling_mean - 2*rolling_std,
    rolling_mean + 2*rolling_std, color='red', alpha=0.1)

    axes[2].set_title(' Adj Closing Price with 30-day Rolling Mean & ±2 Bands')
    axes[2].set_xlabel('Date')
    axes[2].set_ylabel('Adj Close Price')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


    correlation_matrix = df_analysis[['High', 'Low', 'Open', 'Close', 'Volume', 'Adj Close']].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", xticklabels=correlation_matrix.columns, yticklabels=correlation_matrix.columns)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.show()

    scatter_features = ['Open', 'High', 'Low', 'Close', 'Volume','Adj Close']
    pd.plotting.scatter_matrix(df_analysis[scatter_features], alpha=0.2, diagonal='kde', figsize=(12,12))
    plt.title('Scatter Matrix of Selected Features')
    plt.tight_layout()
    plt.show()

    try:
        adj_close_series = df_analysis['Adj Close'].dropna()
        period = 60
        decomposition = seasonal_decompose(adj_close_series, model='additive', period=period, extrapolate_trend='freq')
        fig, axes = plt.subplots(4, 1, figsize=(15, 12))
        fig.suptitle(f'Seasonal Decomposition (Period={period} days)', fontsize=14, y=1.02)
        plot_dates = adj_close_series.index[period-1:]
        axes[0].plot(plot_dates, decomposition.observed[period-1:], color='blue', linewidth=1)
        axes[0].set_title('Observed')
        axes[0].grid(True, alpha=0.3)
        axes[1].plot(plot_dates, decomposition.trend[period-1:], color='green', linewidth=1)
        axes[1].set_title('Trend')
        axes[1].grid(True, alpha=0.3)
        axes[2].plot(plot_dates, decomposition.seasonal[period-1:], color='orange', linewidth=1)
        axes[2].set_title('Seasonal')
        axes[2].grid(True, alpha=0.3)
        axes[3].plot(plot_dates, decomposition.resid[period-1:], color='red', linewidth=1)
        axes[3].set_title('Residual')
        axes[3].grid(True, alpha=0.3)
        axes[3].axhline(0, linestyle='--', color='black', linewidth=1)
        plt.tight_layout()
        plt.show()

        seasonal_std = decomposition.seasonal.std()
        trend_std = decomposition.trend.std()
        print(f"Seasonality Std Dev: {seasonal_std:.4f}")
        print(f"Trend Std Dev: {trend_std:.4f}")
        print(f"Seasonality/Trend ratio: {seasonal_std/trend_std:.4f}")
    except Exception as e:
        print(f"Seasonal decomposition warning: {e}")
        print("Trying with different period...")

def init_data_prep(df,features,Seq,target_idx=5):
    data = df[features].values
    scaler = MinMaxScaler(feature_range=(0,1))
    data_scaled = scaler.fit_transform(data)
    X, y = [], []
    for i in range(Seq, len(data_scaled)):
        X.append(data_scaled[i-Seq:i])
        y.append(data_scaled[i, target_idx])
    X, y = np.array(X), np.array(y)

    train_size = int(0.8*len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test, scaler, features, train_size

def lstm_model(units, Dpr, input_shape):
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss = 'mean_squared_error'
    metrics = ['mae']
    model = Sequential()
    model.add(LSTM(units=units, return_sequences=True, input_shape=input_shape)),BatchNormalization()
    model.add(LSTM(units=units, return_sequences=True, input_shape=input_shape)), LayerNormalization()
    model.add(LSTM(units=units, return_sequences=False, input_shape=input_shape))
    model.add(Dropout(Dpr))
    model.add(Dense(units=1))
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    model.summary()
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)
    ]
    return model, callbacks

def model_train(model, callbacks, X_train, y_train, X_test, y_test, epochs, batch_size):
    history = model.fit(
        X_train, y_train,
        validation_data = (X_test, y_test),
        epochs = epochs,
        batch_size = batch_size,
        callbacks = callbacks,
        verbose = 1
    )

    fig, axes = plt.subplots(1,2, figsize=(14,5))
    axes[0].plot(history.history['loss'], label='Training Loss')
    axes[0].plot(history.history['val_loss'], label='validation loss')
    axes[0].set_title('Training Loss')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].legend()

    axes[1].plot(history.history['mae'], label='training mae')
    axes[1].plot(history.history['val_mae'], label='validation mae')
    axes[1].set_title('Model MAE')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('MAE')
    axes[1].legend()
    plt.show()
    return model

def model_eval(model, df, train_size, Seq, X_test, y_test, scaler, target_idx=5):
    y_pred = model.predict(X_test)

    y_pred_reshaped = np.zeros((len(y_pred), 6))
    y_test_reshaped = np.zeros((len(y_test), 6))
    y_pred_reshaped[:, target_idx] = y_pred.flatten()
    y_test_reshaped[:, target_idx] = y_test
    y_pred_inverted = scaler.inverse_transform(y_pred_reshaped)[:, target_idx]
    y_test_inverted = scaler.inverse_transform(y_test_reshaped)[:, target_idx]

    mse = mean_squared_error(y_test_inverted, y_pred_inverted)
    rsme = np.sqrt(mse)
    mae = mean_absolute_error(y_test_inverted, y_pred_inverted)
    r2 = r2_score(y_test_inverted, y_pred_inverted)
    print(f"Mean squared error: {mse:.4f}")
    print(f"Root Mean Squared Error: {rsme:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"r2 Score: {r2:.4f}")

    plt.figure(figsize=(12,6))
    plt.plot(y_test_inverted, color='blue', label='Actual Stock Price', alpha=0.7)
    plt.plot(y_pred_inverted, color='green', label='Predicted Stock Price', alpha=0.7)
    plt.title('Actual vs Predicted Stock Price')
    plt.xlabel('Time Steps')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

    test_dates = df.index[train_size + Seq:]
    plt.figure(figsize=(12,6))
    plt.plot(test_dates, y_test_inverted, label='Actual Close Price', linewidth=2)
    plt.plot(test_dates, y_pred_inverted, label='Predicted Close Price', linewidth=2, alpha=0.8)
    plt.title('Stock Price Prediction: Actual vs Predicted (With Dates)', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Close Price', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
```
### EDA
<p><img width="1381" height="471" alt="image" src="https://github.com/user-attachments/assets/f1d15841-05a9-47bf-8666-74796148648d" /></p>
<p><img width="494" height="392" alt="image" src="https://github.com/user-attachments/assets/713ee797-243a-44c4-9e33-0c71ee749d4a" /></p>
<p><img width="1034" height="852" alt="image" src="https://github.com/user-attachments/assets/fec19f06-c1b7-43c5-8fd1-5bc7ae0f0761" /></p>

### Training and Testing
```python
X_train, X_test, y_train, y_test, scaler, features, train_size = init_data_prep(stocks_df, ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close'], Seq=60)
model, callbacks = lstm_model(units=4, Dpr=0.2, input_shape=(X_train.shape[1], X_train.shape[2]))
model = model_train(model, callbacks, X_train, y_train, X_test, y_test, epochs = 100, batch_size=32)
model_eval(model, stocks_df, train_size, 60, X_test, y_test, scaler, target_idx=5)
```
<p><img width="809" height="328" alt="image" src="https://github.com/user-attachments/assets/f82a5fc7-b2b1-4a29-98d7-f61af99274be" /></p>
<p><img width="825" height="806" alt="image" src="https://github.com/user-attachments/assets/73313b99-2850-427c-af32-35707a14c19c" /></p>

### Outcomes
- **R²:** 0.73
- Identified lag and tuning opportunities
- Real‑world time‑series forecasting experience
- [Project Link]

---
