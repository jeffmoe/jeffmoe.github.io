"""
model_utils.py
--------------
Online model utilities for next-tick stock price prediction using scikit-learn.

We use an incremental learning approach with SGDRegressor and StandardScaler
so the model can be updated every 30 seconds efficiently.

Target: 1-step-ahead price change (delta) so that prediction = last_price + delta_hat.
Features:
- lag returns (1, 2, 3 lags)
- rolling mean/std of returns (window 5, 10)
- price momentum (price - rolling mean price)
- time-of-day cyclical encoding (sin/cos of seconds since midnight)

Notes:
- The model trains only when enough samples are available (> min_batch).
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


@dataclass
class OnlinePricePredictor:
    random_state: int = 42
    max_iter: int = 1 
    eta0: float = 0.01
    l2: float = 1e-4
    min_batch: int = 20
    feature_lags: int = 3
    price_column: str = "price"

    scaler: StandardScaler = field(default_factory=StandardScaler)
    model: SGDRegressor = field(default_factory=lambda: SGDRegressor(
        loss="squared_error", penalty="l2", alpha=1e-4, learning_rate="constant", eta0=0.01, random_state=42
    ))
    is_fitted: bool = False

    def _feature_engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute features and next-step target.

        Expects df to contain columns: 'timestamp' (ISO) and 'price'.
        Returns a DataFrame with engineered features and target 'y'.
        """
        df = df.copy()
        df = df.sort_values("epoch").reset_index(drop=True)

        df["log_price"] = np.log(df[self.price_column].clip(lower=1e-8))
        df["ret1"] = df["log_price"].diff(1)
        for lag in range(1, self.feature_lags + 1):
            df[f"ret_lag{lag}"] = df["ret1"].shift(lag)

        df["ret_mean5"] = df["ret1"].rolling(5).mean()
        df["ret_std5"] = df["ret1"].rolling(5).std()
        df["ret_mean10"] = df["ret1"].rolling(10).mean()
        df["ret_std10"] = df["ret1"].rolling(10).std()
        df["price_ma10"] = df[self.price_column].rolling(10).mean()
        df["mom10"] = df[self.price_column] - df["price_ma10"]


        sod = (df["epoch"] % 86400).astype(float)
        df["sod_sin"] = np.sin(2 * np.pi * sod / 86400.0)
        df["sod_cos"] = np.cos(2 * np.pi * sod / 86400.0)

        df["y"] = df[self.price_column].diff(-1) * -1

        feature_cols = [
            *(f"ret_lag{lag}" for lag in range(1, self.feature_lags + 1)),
            "ret_mean5", "ret_std5", "ret_mean10", "ret_std10", "mom10",
            "sod_sin", "sod_cos",
        ]
        model_df = df.dropna(subset=feature_cols + ["y"]).copy()
        model_df = model_df.reset_index(drop=True)
        return model_df

    def partial_update(self, df: pd.DataFrame) -> int:
        """Update the model incrementally using available history.

        Returns number of samples used for this update.
        """
        model_df = self._feature_engineer(df)
        if len(model_df) < self.min_batch:
            return 0
        X = model_df.drop(columns=["y", "price", "log_price", "price_ma10"]).select_dtypes(include=[np.number]).values
        y = model_df["y"].values
        self.scaler.partial_fit(X)
        Xs = self.scaler.transform(X)

        if not self.is_fitted:
            self.model.partial_fit(Xs, y)
            self.is_fitted = True
        else:
            self.model.partial_fit(Xs, y)
        return len(model_df)

    def predict_next(self, df: pd.DataFrame) -> tuple[float | None, dict]:
        """Predict next price given the latest observations.

        Returns (predicted_price, debug_info)
        """
        if not self.is_fitted or len(df) < self.feature_lags + 11:
            return None, {"reason": "insufficient_data_or_model_not_fitted"}

        fe = self._feature_engineer(df)
        if fe.empty:
            return None, {"reason": "no_features"}

        last_row = fe.iloc[[-1]]
        X = last_row.drop(columns=["y", "price", "log_price", "price_ma10"]).select_dtypes(include=[np.number]).values
        Xs = self.scaler.transform(X)
        delta_hat = float(self.model.predict(Xs)[0])
        last_price = float(df[self.price_column].iloc[-1])
        return last_price + delta_hat, {"last_price": last_price, "delta_hat": delta_hat}
