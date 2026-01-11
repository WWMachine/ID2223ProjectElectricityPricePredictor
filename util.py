import requests
import requests_cache
import pandas as pd
import openmeteo_requests
from retry_requests import retry
import json
import hopsworks
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
import os
import datetime
import pytz

import matplotlib.pyplot as plt
import matplotlib.dates as mdates


import numpy as np

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from xgboost import XGBRegressor
from xgboost import plot_importance
from sklearn.metrics import mean_squared_error, r2_score



class HopsworksSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        extra='ignore'
    )
    


    HOPSWORKS_API_KEY: SecretStr | None = None

    

def get_historic_price(price_fg, start_date: datetime.date = None, end_date: datetime.date = None):
    AREA = "SE3" 
    if start_date == None: 
        start_date = datetime.date(2023, 12, 17)
        end_date   = datetime.date(2026, 1, 7)
         

    all_data = []

    current = start_date
    while current <= end_date:
        yyyy = current.year
        mmdd = f"{current.month:02d}-{current.day:02d}"
        
        url = f"https://www.elprisetjustnu.se/api/v1/prices/{yyyy}/{mmdd}_{AREA}.json"
        
        resp = requests.get(url, timeout=15)

        daily = resp.json()
            
            
        for entry in daily:
            
            ts = pd.to_datetime(entry["time_start"]) #read prices in swedish time
            
      
            if ts.minute == 0:
                all_data.append({
                    "date": ts,
                    "price_sek_per_kwh": entry["SEK_per_kWh"]
                })
        
        current += datetime.timedelta(days=1)

  
    df = pd.DataFrame(all_data)
    


   
    df["date"] = pd.to_datetime(df["date"], utc=True)

    
    df["date"] = df["date"].dt.tz_convert("Europe/Stockholm")

    
    df["hour"] = df["date"].dt.hour
    df["day"] = df["date"].dt.day
    df["month"] = df["date"].dt.month


    df["date"] = df["date"].dt.tz_convert("UTC")# convert to utc for feature store

    
    if price_fg == None:
        df.sort_values("date").reset_index(drop=True)
        
        df["price_24h_ago"] = df["price_sek_per_kwh"].shift(24)
        
    else:
        query = price_fg.select_all().filter(price_fg['date']>= start_date-datetime.timedelta(days=31))
        df_recent = query.read()
        df_recent = df_recent.sort_values('date')
        
        df_recent["date"] = pd.to_datetime(df_recent["date"], utc=True)
      
        new_timestamps = df["date"]

        combined = pd.concat([df_recent, df], ignore_index=True)

        combined = combined.sort_values("date").drop_duplicates('date')
        
        combined["price_24h_ago"] = combined["price_sek_per_kwh"].shift(24)
        
        df = combined[combined["date"].isin(new_timestamps)].reset_index(drop=True)
        
        


    return df


def connect_to_hopsworks_project():
    project = hopsworks.login()
    return project


def get_historical_weather(city, start_date,  end_date, latitude, longitude):
    cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
    retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
    openmeteo = openmeteo_requests.Client(session = retry_session)

    
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = { # read one day before start and one day after end in the range to cmake sure values exist for the edge cases when handling utc conversion
        "latitude": latitude,
        "longitude": longitude,
        "start_date": (pd.to_datetime(start_date) - pd.Timedelta(days=1)).strftime('%Y-%m-%d'),
        "end_date": (pd.to_datetime(end_date) + pd.Timedelta(days=1)).strftime('%Y-%m-%d'),
        "hourly": ["temperature_2m", "precipitation", "wind_speed_10m", "cloud_cover"],
    }
    responses = openmeteo.weather_api(url, params=params)

    
    response = responses[0]


    
    hourly = response.Hourly()
    hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
    hourly_precipitation = hourly.Variables(1).ValuesAsNumpy()

    hourly_wind_speed_10m = hourly.Variables(2).ValuesAsNumpy()

    hourly_cloud_cover = hourly.Variables(3).ValuesAsNumpy()

    hourly_data = {"date": pd.date_range(
        start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
        end =  pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
        freq = pd.Timedelta(seconds = hourly.Interval()),
        inclusive = "left"
    )}

    hourly_data[f"temperature_2m_{city}"] = hourly_temperature_2m
    hourly_data[f"precipitation_{city}"] = hourly_precipitation
    hourly_data[f"wind_speed_10m_{city}"] = hourly_wind_speed_10m
    hourly_data[f"cloud_cover_{city}"] = hourly_cloud_cover

    hourly_dataframe = pd.DataFrame(data = hourly_data)
    hourly_dataframe = hourly_dataframe.dropna()
    

    return hourly_dataframe





def create_fs(project):
    fs = project.get_feature_store() 
    return fs

def create_weather_fg(fs, city):
    weather_fg = fs.get_or_create_feature_group(
    name=f"weather_{city.lower()}",
    description='Weather characteristics of each day',
    version=1,
    primary_key=['date'],
    event_time="Date",
    expectation_suite=None
    ) 
    return weather_fg

def insert_wdf_fg(weather_fg,weather_df, city):
    weather_fg.insert(weather_df, wait=True)
    weather_fg.update_feature_description("date", "Time of measurement of weather")
    weather_fg.update_feature_description(f"temperature_2m_{city}", "Temperature in Celsius")
    weather_fg.update_feature_description(f"precipitation_{city}", "Precipitation (rain/snow) in mm")
    weather_fg.update_feature_description(f"wind_speed_10m_{city}", "Wind speed at 10m abouve ground")
    weather_fg.update_feature_description(f"cloud_cover_{city}","How cloudy the overcast is")
    return weather_fg


def create_price_fg(fs):
    price_fg = fs.get_or_create_feature_group(
    name=f"price_swedene3",
    description='Hourly Price each day',
    version=1,
    primary_key=['date'],
    event_time="date",
    expectation_suite=None
    ) 
    return price_fg

def insert_price_fg(price_fg,price_df):
    price_fg.insert(price_df, wait=True)
    price_fg.update_feature_description("date", "Time of Price")
    price_fg.update_feature_description("price_sek_per_kwh", "Price")
    
    price_fg.update_feature_description("price_24h_ago", "Electricity price 24 hours ago")
    
    return price_fg


def create_json_secret(city,latitude,longitude):
    secrets = hopsworks.get_secrets_api()
    dict_obj = {
        "city": city,
        "latitude": latitude,
        "longitude": longitude
    }
    
    
    str_dict = json.dumps(dict_obj)
    
    
    secret = secrets.get_secret(f"SENSOR_LOCATION_JSON_{city.lower()}")
    if secret is not None:
        secret.delete()
        print("Replacing existing SENSOR_LOCATION_JSON")
    
    secrets.create_secret(f"SENSOR_LOCATION_JSON_{city.lower()}", str_dict)
    


api_key = os.getenv("HOPSWORKS_API_KEY")

if not api_key:
    
    settings = HopsworksSettings(_env_file=".env")
    api_key = settings.HOPSWORKS_API_KEY.get_secret_value()

os.environ["HOPSWORKS_API_KEY"] = api_key





###### PART 2 DAILY ######

def forecast_weather(latitude, longitude, city):
    cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
    retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
    openmeteo = openmeteo_requests.Client(session = retry_session)

    
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": ["temperature_2m", "precipitation", "wind_speed_10m", "cloud_cover"],
        "past_days": 1, # need to get 23:00 utc because that is 00:00 swedish time
        "forecast_days": 1,
    }
    responses = openmeteo.weather_api(url, params=params)

    
    response = responses[0]
    print(f"Coordinates: {response.Latitude()}°N {response.Longitude()}°E")
    print(f"Elevation: {response.Elevation()} m asl")
    print(f"Timezone difference to GMT+0: {response.UtcOffsetSeconds()}s")

    
    hourly = response.Hourly()
    hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
    hourly_precipitation = hourly.Variables(1).ValuesAsNumpy()
    hourly_wind_speed_10m = hourly.Variables(2).ValuesAsNumpy()
    hourly_cloud_cover = hourly.Variables(3).ValuesAsNumpy()


    hourly_data = {"date": pd.date_range(
        start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
        end =  pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
        freq = pd.Timedelta(seconds = hourly.Interval()),
        inclusive = "left"
    )}

    hourly_data[f"temperature_2m_{city}"] = hourly_temperature_2m
    hourly_data[f"precipitation_{city}"] = hourly_precipitation
    hourly_data[f"wind_speed_10m_{city}"] = hourly_wind_speed_10m
    hourly_data[f"cloud_cover_{city}"] = hourly_cloud_cover

    
    hourly_dataframe = pd.DataFrame(data = hourly_data) # remove the unnecessary data so we only have the 24 entries corresponding to the swedish day

    hourly_dataframe["date_swe"] = hourly_dataframe["date"].dt.tz_convert("Europe/Stockholm")

    today_swe = pd.Timestamp.now(tz="Europe/Stockholm").date()
    

    df_filtered = hourly_dataframe[hourly_dataframe["date_swe"].dt.date == today_swe].copy()
    
    df_filtered = df_filtered.drop(columns=["date_swe"]).reset_index(drop=True)
    
    
    return df_filtered

def get_fgs(fs,city1,city2):
    
    price_fg = fs.get_feature_group(
        name="price_swedene3",
        version=1,
    )
    weather_1_fg = fs.get_feature_group(
        name=f"weather_{city1.lower()}",
        version=1,
    )
    weather_2_fg = fs.get_feature_group(
        name=f"weather_{city2.lower()}",
        version=1,
    )
    return price_fg, weather_1_fg, weather_2_fg

def weather_insert(weather_fg, daily_df):
    weather_fg.insert(daily_df, wait=True)

    return weather_fg

def price_insert(price_fg,latest_price_df):
    # Insert new data
    price_fg.insert(latest_price_df, wait=True)

    return price_fg







###### PART 3 MODEL ######




def set_selected_features(electricity_fg, weather_1_fg, weather_2_fg):
    
    
    
    selected_features = (
    electricity_fg
    .select([
    "price_sek_per_kwh",
    "hour",
    "day", 
    "month", 
    "date", 
    
    "price_24h_ago" 
    
    ])
    .join(weather_1_fg.select_features(), on=['date'])
    .join(weather_2_fg.select_features(), on=['date'])
    )    
    return selected_features

def create_fv(fs,selected_features):
    feature_view = fs.get_or_create_feature_view(
        name=f"fv",
        description="weather features with electricity price as the target",
        version=1,
        labels=['price_sek_per_kwh'],
        query=selected_features,
    )
    return feature_view

def set_training_data(fv):


    start_datetime = datetime.datetime(2025, 10, 1)

    X_tr, X_ts, y_tr, y_ts = fv.train_test_split(test_start = start_datetime)

    

    return X_tr, X_ts, y_tr, y_ts

def get_features(X_tr,X_ts):
    X_tr_feat = X_tr.drop(columns = ['date'])
    X_ts_feat = X_ts.drop(columns = ['date'])
    return X_tr_feat,X_ts_feat
    

def model_creation(X_tr, y_tr):

    xgb = XGBRegressor()
    xgb.fit(X_tr,y_tr)
    return xgb 

def pred_eval(model, X_ts, y_ts):
    
    y_pr = model.predict(X_ts)

    y_ts_prep = y_ts.iloc[:,0]
    mse = mean_squared_error(y_ts_prep, y_pr)
    print(f'The MSE is: {mse}')

    r2 = r2_score(y_ts_prep, y_pr)
    print(f'The R2-score is: {r2}')

    return y_pr, mse, r2

def df_creation(y_ts, y_pr, X_ts):
    df = y_ts
    df['pred_price'] = y_pr
    df['date'] = X_ts['date']
    df = df.sort_values(by=['date'])
    df.head(5)
    return df

def create_dirs():
    
    model_dir = f"docs"
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    images_dir = model_dir + "/assets/img"
    if not os.path.exists(images_dir):
        os.mkdir(images_dir)
    return model_dir, images_dir



def importance_plot(xgb_regressor,images_dir):
    
    plot_importance(xgb_regressor)
    feature_importance_path = images_dir + f"/feature_importance.png"
    plt.savefig(feature_importance_path)
    plt.show()

def plot_train(region: str, df: pd.DataFrame, file_path: str, hindcast=False):

    fig, ax = plt.subplots(figsize=(14, 6))


    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    
    ax.plot(
        df["date"],
        df["pred_price"],
        label="Predicted Electricity Price",
        color="red",
        linewidth=1.8
    )

    if hindcast:
        ax.plot(
            df["date"],
            df["price_sek_per_kwh"],
            label="Actual Price",
            color="black",
            linewidth=1.8,
            alpha=0.8
        )

    
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=3)) 
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

    ax.set_xlabel("Date")
    ax.set_ylabel("SEK / kwh")
    ax.set_title(f"Hourly Electricity Price Forecast {region}")

    plt.xticks(rotation=45)
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(file_path, dpi=300)
    plt.close()

    return file_path



def save_model(model, model_dir):
    
    booster = model.get_booster()
    booster.save_model(model_dir + "/model.json")
    

def saving_model_hops( fv, mse, r2, project, dir):
    model_reg = project.get_model_registry()

    model_elprice = model_reg.python.create_model(
        name="model_elprice",
        metrics= {"MSE": str(mse), "R2": str(r2)},
        feature_view = fv,
        description = "Predictions of electricity prices for Sweden Area SE3."
    )
    model_elprice.save(dir)




###### PART 4 INFERENCE ######

def get_model(project):
    mr = project.get_model_registry()
    
    retrieved_model = mr.get_model(
        name=f"model_elprice",
        version=1,
    )
    
    fv = retrieved_model.get_feature_view()
    
    saved_model_dir = retrieved_model.download()

    xgboost_model = XGBRegressor()

    xgboost_model.load_model(saved_model_dir + f"/model.json")

    return xgboost_model, fv

def get_forecast_weather(fs,today,city1, city2):
    today = today - datetime.timedelta(hours=1)
    end_date = today + datetime.timedelta(hours=24) 
    


    weather_fg_1 = fs.get_feature_group(
        name=f"weather_{city1.lower()}",
        version=1,
    )

    batch_data_1 = weather_fg_1.filter(
        (weather_fg_1.date >= today) & (weather_fg_1.date <= end_date)
    ).read()
    
    weather_fg_2 = fs.get_feature_group(
        name=f"weather_{city2.lower()}",
        version=1,
    )
    batch_data_2 = weather_fg_2.filter(
        (weather_fg_2.date >= today) & (weather_fg_2.date <= end_date)
    ).read()
    

    combined_data = pd.merge(batch_data_1, batch_data_2, on="date", how="inner")
    with pd.option_context('display.max_rows', None):
        print("COMBINED", combined_data["date"])

    return combined_data.sort_values("date").reset_index(drop=True)

def fill_features(hour,batch_data,prev_data):
    hour_swe = hour.astimezone(pytz.timezone("Europe/Stockholm"))
    
    mask = batch_data["date"] == hour
    
    batch_data.loc[mask, "hour"] = hour_swe.hour
    batch_data.loc[mask, "day"] = hour_swe.day
    batch_data.loc[mask, "month"] = hour_swe.month

    batch_data.loc[mask, "price_24h_ago"] = prev_data["price_sek_per_kwh"].iloc[-24]
    
    return batch_data

def add_first_price_features(fs,batch_data):

    
    first_hour = (datetime.datetime.now(tz=pytz.UTC)).replace(hour=23, minute=0, second=0, microsecond=0) - datetime.timedelta(days=1) # 1 day is default
    
    price_group = fs.get_feature_group(name=f"price_swedene3", version=1)
    first_init_roll = price_group.filter(price_group.date >= first_hour -  datetime.timedelta(3)).read()
    
    
    first_init_roll = first_init_roll.sort_values("date").reset_index(drop=True)
    batch_data = fill_features(first_hour,batch_data,first_init_roll)
    
    

    return batch_data, first_hour, first_init_roll


def predictions(hour,batch_data,model,first_init_roll):
    

    batch_data["date"] = pd.to_datetime(batch_data["date"], utc=True)
    time_obj = pd.to_datetime(hour, utc=True)
    print("LEN BATCH PRED", len(batch_data))
    
    for _ in range(len(batch_data)):
        
    
        mask = batch_data["date"] == time_obj
        
        
        if not mask.any():
            break
        
        
        
        X = batch_data.loc[mask,["hour", "day", "month",  
        "price_24h_ago", "temperature_2m_stockholm", "precipitation_stockholm", 
                                 "wind_speed_10m_stockholm", "cloud_cover_stockholm", "temperature_2m_goteborg", "precipitation_goteborg", 
                                 "wind_speed_10m_goteborg", "cloud_cover_goteborg"]]
    
        y_pred = model.predict(X)[0]
        
        
        batch_data.loc[mask, "pred_price"] = y_pred
        
        new_data = {
            "date": time_obj,
            "price_sek_per_kwh": y_pred,
        }
        
        
    
        first_init_roll = pd.concat([first_init_roll, pd.DataFrame([new_data])], ignore_index=True)
        first_init_roll = first_init_roll.sort_values("date").reset_index(drop=True)
        time_next = time_obj + datetime.timedelta(hours=1)
    
        if(batch_data["date"]== time_next).any():
            batch_data = fill_features(time_next,batch_data,first_init_roll)
            
        time_obj = time_next
        
    batch_data['days_before_forecast_day'] = (np.arange(len(batch_data)) // 24 + 1).astype(np.int32)
    
    print(len(batch_data))
    print(batch_data["pred_price"])
    print(batch_data["date"])
    return batch_data



def plot_price_forecast(region: str, df: pd.DataFrame, file_path: str, hindcast=False):
    fig, ax = plt.subplots(figsize=(14, 6))

    df = df.copy()
    
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df["date"] = df["date"].dt.tz_convert("Europe/Stockholm").dt.tz_localize(None)
    df = df.sort_values("date")

    with pd.option_context('display.max_rows', None):
        print(df["date"])

    
    ax.plot(df["date"], df["pred_price"], label="Predicted Electricity Price", color="red", linewidth=1.8)
    if hindcast and "price_sek_per_kwh" in df.columns:
        ax.plot(df["date"], df["price_sek_per_kwh"], label="Actual Price", color="black", linewidth=1.8, alpha=0.8)

    
    ax.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 6, 12, 18]))
    
    
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))

    
    ax.xaxis.set_minor_locator(mdates.HourLocator(interval=1))

    
    ax.set_xlim(df["date"].min(), df["date"].max())

    
    ax.grid(True, which='major', color='gray', linestyle='-', alpha=0.4)
    ax.grid(True, which='minor', color='gray', linestyle='--', alpha=0.2)

    ax.set_xlabel("Swedish Local Time")
    ax.set_ylabel("SEK / kwh")
    
    ax.set_title(f"Hourly Electricity Price {'Hindcast' if hindcast else 'Forecast'} {region}")

    plt.xticks(rotation=45)
    ax.legend()

    plt.tight_layout()
    plt.savefig(file_path, dpi=300)
    plt.close()

    return file_path

def monitoring_fg(batch_data,fs):
    
    monitor_fg = fs.get_or_create_feature_group(
        name=f"price_prediction_monitoring",
        description='Electricity price real vs predicted comparison',
        version=1,
        primary_key=['date','days_before_forecast_day'],
        event_time="date"
    )
    
    monitor_fg.insert(batch_data, wait=True)

    return monitor_fg

def monitor_df(monitor_fg):
    import time
    time.sleep(5)
    monitoring_df = monitor_fg.filter(monitor_fg.days_before_forecast_day == 1).read()
    
    print(f"DEBUG: monitoring_df rows fetched: {len(monitoring_df)}")
    if not monitoring_df.empty:
        print(f"DEBUG: Latest date in monitor: {monitoring_df['date'].max()}")
    return monitoring_df

def elprice_pred_df(fs):
    electricity_fg = fs.get_feature_group(name="price_swedene3", version=1)
    electricity_df = electricity_fg.read()

    return electricity_fg, electricity_df

def hindcast(electricity_price_df, monitoring_df):
    outcome_df = electricity_price_df[['date', 'price_sek_per_kwh']].sort_values(by=['date'])
    preds_df =  monitoring_df[['date', 'pred_price']].sort_values(by=['date'])
    
    with pd.option_context('display.max_rows', None):
        print("PREDS", preds_df)
    hindcast_df = pd.merge(preds_df, outcome_df, on="date")
    hindcast_df = hindcast_df.sort_values(by=['date'])
    with pd.option_context('display.max_rows', None):
    
        print("HINDCAST DF", hindcast_df)
    
    return hindcast_df



def upload_to_hops(project, today,pred_path,hind_path):
    dataset_api = project.get_dataset_api()
    str_today = today.strftime("%Y-%m-%d")
    if dataset_api.exists("Resources/SE3") == False:
        dataset_api.mkdir("Resources/SE3")
    dataset_api.upload(pred_path, f"Resources/SE3/forecast_{str_today}", overwrite=True)
    dataset_api.upload(hind_path, f"Resources/SE3/hindcast_{str_today}", overwrite=True)



