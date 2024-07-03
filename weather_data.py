import openmeteo_requests
import requests_cache
import pandas as pd
import numpy as np
from retry_requests import retry
from matplotlib.figure import Figure
from datetime import datetime


def get_historical_weather(start_date, end_date):
    # Setup the Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    # Make sure all required weather variables are listed here
    # The order of variables in hourly or daily is important to assign them
    # correctly below
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": 55.7522,
        "longitude": 37.6156,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "hourly": ["temperature_2m", "relative_humidity_2m", "precipitation",
                   "weather_code", "surface_pressure"],
        "timezone": "Europe/Moscow"
    }
    responses = openmeteo.weather_api(url, params=params)

    # Process first location.
    # Add a for-loop for multiple locations or weather models
    response = responses[0]

    # Process hourly data.
    # The order of variables needs to be the same as requested.
    hourly = response.Hourly()
    hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
    hourly_relative_humidity_2m = hourly.Variables(1).ValuesAsNumpy()
    hourly_precipitation = hourly.Variables(2).ValuesAsNumpy()
    hourly_weather_code = hourly.Variables(3).ValuesAsNumpy()
    hourly_surface_pressure = hourly.Variables(4).ValuesAsNumpy()

    hourly_data = {"datetime": pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left"
    )}
    hourly_data["temperature_2m"] = hourly_temperature_2m
    hourly_data["relative_humidity_2m"] = hourly_relative_humidity_2m
    hourly_data["precipitation"] = hourly_precipitation
    hourly_data["weather_code"] = hourly_weather_code
    hourly_data["surface_pressure"] = hourly_surface_pressure

    return pd.DataFrame(data=hourly_data)


def max_diff_in_window(dataframe, column='surface_pressure_mmHg', window=48):
    dataframe[column+'max_diff_backward'] = None
    dataframe[column+'max_diff_forward'] = None
    for i in range(len(dataframe)):
        current_point = dataframe.loc[i, column]

        window_backward = current_point - dataframe.loc[i-window:i, column]
        max_diff_backward = window_backward[
            abs(window_backward) == max(abs(window_backward))].values[0]
        dataframe.loc[i, column+'max_diff_backward'] = max_diff_backward

        window_forward = current_point - dataframe.loc[i:i+window, column]
        max_diff_forward = window_forward[
            abs(window_forward) == max(abs(window_forward))].values[0]
        dataframe.loc[i, column+'max_diff_forward'] = max_diff_forward


def get_peaks(row, threshold=3):
    if (
        row['surface_pressure_mmHgmax_diff_backward'] >= threshold and
        row['surface_pressure_mmHgmax_diff_forward'] >= threshold
    ):
        return row['surface_pressure_mmHg']
    else:
        return None


def get_valleys(row, threshold=3):
    if (
        row['surface_pressure_mmHgmax_diff_backward'] <= -threshold and
        row['surface_pressure_mmHgmax_diff_forward'] <= -threshold
    ):
        return row['surface_pressure_mmHg']
    else:
        return None


def get_rapid_ups(row, threshold=3):
    if row['surface_pressure_mmHgmax_diff_forward'] <= -threshold:
        return row['surface_pressure_mmHg']
    else:
        return None


def get_rapid_downs(row, threshold=3):
    if row['surface_pressure_mmHgmax_diff_forward'] >= threshold:
        return row['surface_pressure_mmHg']
    else:
        return None


def get_surface_pressure_changes(dataframe, threshold=3):
    dataframe['peak'] = dataframe.apply(get_peaks, axis=1,
                                        threshold=threshold)
    dataframe['valley'] = dataframe.apply(get_valleys, axis=1,
                                          threshold=threshold)
    dataframe['rapid_up'] = dataframe.apply(get_rapid_ups, axis=1,
                                            threshold=threshold)
    dataframe['rapid_down'] = dataframe.apply(get_rapid_downs, axis=1,
                                              threshold=threshold)


def get_surface_pressure_analysis(dataframe, window=48, threshold=3):
    CONVERT_hPa_TO_mmHg = 0.7500637554

    dataframe.dropna(inplace=True)
    dataframe['surface_pressure_mmHg'] = (
        dataframe['surface_pressure'] * CONVERT_hPa_TO_mmHg)
    max_diff_in_window(dataframe, column='surface_pressure_mmHg',
                       window=window)
    get_surface_pressure_changes(dataframe, threshold=threshold)
    return dataframe


def get_surface_pressure_flags(dataframe):
    dataframe['date'] = dataframe['datetime'].dt.date
    return dataframe[
        ['date', 'peak', 'valley', 'rapid_up', 'rapid_down']].groupby(
        'date').agg(lambda x: 0 if np.isnan(max(x)) else 1).reset_index()


def get_graph(dataframe):
    fig = Figure(figsize=(8, 4))
    ax = fig.subplots()
    ax.plot(dataframe['datetime'], dataframe['surface_pressure_mmHg'],
            linestyle='-')
    ax.plot(dataframe['datetime'], dataframe['peak'],
            linestyle='-', color='red')
    ax.plot(dataframe['datetime'], dataframe['valley'],
            linestyle='-', color='red')

    ax.set_title('Surface pressure, mmHg')
    ax.set_xlabel('Date')
    ax.set_ylabel('Surface pressure, mmHg')
    major_tick = dataframe['datetime'].dt.date.unique()
    ax.set_xticks(major_tick)
    ax.set_xticklabels(major_tick, rotation=90)
    ax.set_xlim((min(dataframe['datetime']), max(dataframe['datetime'])))

    ax.grid(True, alpha=0.2)
    return fig
