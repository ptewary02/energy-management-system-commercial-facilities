"""
forecast_engine.py
-------------------
Reusable SARIMAX forecasting function for the energy management backend.

This wraps everything from the notebook exploration into one function
your Flask/FastAPI /commands endpoint can call directly.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResults


def build_exog(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Builds the exogenous regressor(s) for a given datetime index.
    Currently just an is_weekend flag - add more signals here later
    (e.g. holiday flag) as your project grows."""
    is_weekend = (index.dayofweek >= 5).astype(int)
    return pd.DataFrame({'is_weekend': is_weekend}, index=index)


def train_zone_model(series: pd.Series, order=(2, 0, 0), seasonal_order=(2, 1, 0, 24)):
    """Trains a SARIMAX model for one zone's historical data and returns the fitted result.

    series: a pandas Series of hourly kW readings, indexed by Datetime, with .index.freq set.
    """
    if series.index.freq is None:
        series = series.asfreq('h')

    exog = build_exog(series.index)
    model = SARIMAX(
        series,
        exog=exog,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    return model.fit(disp=False)


def forecast_next_24h(fit_result: SARIMAXResults, last_timestamp: pd.Timestamp) -> pd.DataFrame:
    """Given a fitted model and the last known timestamp, forecasts the next 24 hours.

    Returns a DataFrame with columns: predicted_kW, lower_95, upper_95
    """
    future_index = pd.date_range(start=last_timestamp + pd.Timedelta(hours=1), periods=24, freq='h')
    future_exog = build_exog(future_index)

    forecast = fit_result.get_forecast(steps=24, exog=future_exog)
    predicted_mean = forecast.predicted_mean
    conf_int = forecast.conf_int(alpha=0.05)

    result = pd.DataFrame({
        'predicted_kW': predicted_mean.values,
        'lower_95': conf_int.iloc[:, 0].values,
        'upper_95': conf_int.iloc[:, 1].values,
    }, index=future_index)

    return result


if __name__ == "__main__":
    # Quick smoke test using the synthetic dataset
    df = pd.read_csv('/mnt/user-data/outputs/synthetic_energy_dataset.csv', parse_dates=['Datetime'])
    df = df.set_index('Datetime')
    df.index.freq = 'h'

    zone_col = 'Zone1_Lighting_kW'
    training_data = df[zone_col].iloc[-24*30:]  # last 30 days

    print(f"Training model on {zone_col}...")
    fit = train_zone_model(training_data)

    forecast_df = forecast_next_24h(fit, training_data.index[-1])
    print("\nNext 24h forecast:")
    print(forecast_df.round(2).to_string())