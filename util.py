
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
from datetime import date, timedelta
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.patches import Patch
import datetime
import pytz
import numpy as np

#to get historic

mode_select = 0

class HopsworksSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        extra='ignore'
    )
    

    # For hopsworks.login(), set as environment variables if they are not already set as env variables
    HOPSWORKS_API_KEY: SecretStr | None = None

    

def get_historic_price(price_fg, start_date: date = None, end_date: date = None):
    AREA = "SE3" 
    if start_date == None: 
        start_date = date(2023, 12, 17)
        end_date   = date(2026, 1, 5)
         

    all_data = []

    current = start_date
    while current <= end_date:
        yyyy = current.year
        mmdd = f"{current.month:02d}-{current.day:02d}"
        
        url = f"https://www.elprisetjustnu.se/api/v1/prices/{yyyy}/{mmdd}_{AREA}.json"
        
        resp = requests.get(url, timeout=15)

        daily = resp.json()
            
            
        for entry in daily:
            
            ts = pd.to_datetime(entry["time_start"], utc=True)
            
            # Keep only full hours (minute and second = 0)
            if ts.minute == 0:
                all_data.append({
                    "date": ts,
                    "price_sek_per_kwh": entry["SEK_per_kWh"]
                })
        
        current += timedelta(days=1)

    # Turn into DataFrame
    df = pd.DataFrame(all_data)
    


    # Convert to datetime
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df["hour"] = df["date"].dt.hour
    df["day"] = df["date"].dt.day  
    df["month"] = df["date"].dt.month

    df["date"] = df["date"].dt.tz_convert("Europe/Stockholm")
    if price_fg == None:
        df.sort_values("date").reset_index(drop=True)
        #df["price_1h_ago"]  = df["price_sek_per_kwh"].shift(1)
        #df["price_2h_ago"]  = df["price_sek_per_kwh"].shift(2)
        #df["price_3h_ago"]  = df["price_sek_per_kwh"].shift(3)
        df["price_24h_ago"] = df["price_sek_per_kwh"].shift(24)
        #df["price_48h_ago"] = df["price_sek_per_kwh"].shift(48)
        #df["price_72h_ago"] = df["price_sek_per_kwh"].shift(72)
        #df["price_avg_same_hour_last_week"] = (
        #df.groupby(df["date"].dt.hour)["price_sek_per_kwh"]
        #.transform(lambda x: x.shift(1).rolling(7).mean())
        #)
        #df["price_avg_same_hour_last_month"] = (
        #    df.groupby(df["date"].dt.hour)["price_sek_per_kwh"]
        #    .transform(lambda x: x.shift(1).rolling(30).mean())
        #)
        #df["price_last_168h_avg"] = df["price_sek_per_kwh"].shift(1).rolling(168).mean()
    else:
        query = price_fg.select_all().filter(price_fg['date']>= start_date-timedelta(days=31))
        df_recent = query.read()
        df_recent = df_recent.sort_values('date')
        combined = pd.concat([df_recent, df], ignore_index=True)

        # Sort by date
        combined = combined.sort_values("date")

        # Calculate lag features on the combined dataframe
        #combined["price_1h_ago"]  = combined["price_sek_per_kwh"].shift(1)
        #combined["price_2h_ago"]  = combined["price_sek_per_kwh"].shift(2)
        #combined["price_3h_ago"]  = combined["price_sek_per_kwh"].shift(3)
        combined["price_24h_ago"] = combined["price_sek_per_kwh"].shift(24)
        #combined["price_48h_ago"] = combined["price_sek_per_kwh"].shift(48)
        #combined["price_72h_ago"] = combined["price_sek_per_kwh"].shift(72)

        # Rolling/average features
        combined["date"] = pd.to_datetime(combined["date"], utc=True)  # convert to datetime if not already
        
        #combined["price_avg_same_hour_last_week"] = (
        #    combined.groupby(combined["date"].dt.hour)["price_sek_per_kwh"]
        #    .transform(lambda x: x.shift(1).rolling(7).mean())
        #)

        #combined["price_avg_same_hour_last_month"] = (
        #    combined.groupby(combined["date"].dt.hour)["price_sek_per_kwh"]
        #    .transform(lambda x: x.shift(1).rolling(30).mean())
        #)

        #combined["price_last_168h_avg"] = combined["price_sek_per_kwh"].shift(1).rolling(168).mean()

        # Now extract only the new rows
        
        df = combined.iloc[len(df_recent):].reset_index(drop=True)

    return df
#to get hourly (once per day?)

def connect_to_hopsworks_project():
    project = hopsworks.login()
    return project

# weather
def get_historical_weather(city, start_date,  end_date, latitude, longitude):
    cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
    retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
    openmeteo = openmeteo_requests.Client(session = retry_session)

    # Make sure all required weather variables are listed here
    # The order of variables in hourly or daily is important to assign them correctly below
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ["temperature_2m", "precipitation", "wind_speed_10m", "cloud_cover"],
    }
    responses = openmeteo.weather_api(url, params=params)

    # Process first location. Add a for-loop for multiple locations or weather models
    response = responses[0]
    print(f"Coordinates: {response.Latitude()}°N {response.Longitude()}°E")
    print(f"Elevation: {response.Elevation()} m asl")
    print(f"Timezone difference to GMT+0: {response.UtcOffsetSeconds()}s")

    # Process hourly data. The order of variables needs to be the same as requested.
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
    #price_fg.update_feature_description("price_1h_ago", "Electricity price 1 hour ago")
    #price_fg.update_feature_description("price_2h_ago", "Electricity price 2 hours ago")
    #price_fg.update_feature_description("price_3h_ago", "Electricity price 3 hours ago")
    price_fg.update_feature_description("price_24h_ago", "Electricity price 24 hours ago")
    #price_fg.update_feature_description("price_48h_ago", "Electricity price 48 hours ago")
    #price_fg.update_feature_description("price_72h_ago", "Electricity price 72 hours ago") 
    #price_fg.update_feature_description(
    #    "price_avg_same_hour_last_week",
    #    "Average electricity price for the same hour over the last 7 days"
    #)
    #price_fg.update_feature_description(
    #    "price_avg_same_hour_last_month",
    #    "Average electricity price for the same hour over the last 30 days"
    #)
    #price_fg.update_feature_description(
    #    "price_last_168h_avg",
    #    "Average electricity price over the last week"
    #)
    return price_fg


def create_json_secret(city,latitude,longitude):
    secrets = hopsworks.get_secrets_api()
    dict_obj = {
        "city": city,
        "latitude": latitude,
        "longitude": longitude
    }
    
    # Convert the dictionary to a JSON string
    str_dict = json.dumps(dict_obj)
    
    # Replace any existing secret with the new value
    secret = secrets.get_secret(f"SENSOR_LOCATION_JSON_{city.lower()}")
    if secret is not None:
        secret.delete()
        print("Replacing existing SENSOR_LOCATION_JSON")
    
    secrets.create_secret(f"SENSOR_LOCATION_JSON_{city.lower()}", str_dict)
    


api_key = os.getenv("HOPSWORKS_API_KEY")

if not api_key:
    # Fallback to .env for local development
    settings = HopsworksSettings(_env_file=".env")
    api_key = settings.HOPSWORKS_API_KEY.get_secret_value()

os.environ["HOPSWORKS_API_KEY"] = api_key


if mode_select == 1:
    project = connect_to_hopsworks_project()

    fs = create_fs(project)
    
    cities = {"stockholm":[59.3294,18.0687], "goteborg":[57.7072,11.9668]}

    for key in cities.keys():
        historic = get_historical_weather(key, "2023-12-17", "2026-01-05", cities[key][0], cities[key][1])
        weather_fg = create_weather_fg(fs, key)
        insert_wdf_fg(weather_fg, historic,key)
    price_df = get_historic_price(None)
    price_fg = create_price_fg(fs)
    insert_price_fg(price_fg, price_df)


###### PART 2 DAILY ######

def forecast_weather(latitude, longitude, city):
    cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
    retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
    openmeteo = openmeteo_requests.Client(session = retry_session)

    # Make sure all required weather variables are listed here
    # The order of variables in hourly or daily is important to assign them correctly below
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": ["temperature_2m", "precipitation", "wind_speed_10m", "cloud_cover"],
        "forecast_days": 1,
    }
    responses = openmeteo.weather_api(url, params=params)

    # Process first location. Add a for-loop for multiple locations or weather models
    response = responses[0]
    print(f"Coordinates: {response.Latitude()}°N {response.Longitude()}°E")
    print(f"Elevation: {response.Elevation()} m asl")
    print(f"Timezone difference to GMT+0: {response.UtcOffsetSeconds()}s")

    # Process hourly data. The order of variables needs to be the same as requested.
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
    return hourly_dataframe

def get_fgs(fs,city1,city2):
    # Retrieve feature groups
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

if mode_select == 2:

    project = connect_to_hopsworks_project()

    fs = create_fs(project)

    cities = {"stockholm":[59.3294,18.0687], "goteborg":[57.7072,11.9668]}

    keys = list(cities.keys())

    price_fg, weather_1_fg, weather_2_fg = get_fgs(fs,keys[0],keys[1])

    weather_df_1 = forecast_weather(cities[keys[0]][0],cities[keys[0]][1],keys[0])
    weather_df_2 = forecast_weather(cities[keys[1]][0],cities[keys[1]][1],keys[1])




    today = datetime.datetime.now()
    yesterday = today - timedelta(days=1)



    price_df = get_historic_price(price_fg,yesterday,yesterday)

    weather_insert(weather_1_fg,weather_df_1)
    weather_insert(weather_2_fg,weather_df_2)

    price_insert(price_fg,price_df)





###### PART 3 MODEL ###### NOTE TEST WITHOUT 1 2 3 H ago
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from xgboost import XGBRegressor
from xgboost import plot_importance
from sklearn.metrics import mean_squared_error, r2_score



def set_selected_features(electricity_fg, weather_1_fg, weather_2_fg):
    # Select features for training data.
    
    
    selected_features = (
    electricity_fg
    .select([
    "price_sek_per_kwh",
    "hour",
    "day", 
    "month", 
    "date", 
    #"price_1h_ago", 
    #"price_2h_ago", 
    #"price_3h_ago", 
    "price_24h_ago", 
    #"price_48h_ago", 
    #"price_72h_ago",
    #"price_avg_same_hour_last_week",
    #"price_avg_same_hour_last_month",
    #"price_last_168h_avg"
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

    start_date = "2025-10-01" # CHANGE THIS BACK
    start_datetime = datetime.strptime(start_date, "%Y-%m-%d")

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
    # Creating a directory for the model artifacts if it doesn't exist
    model_dir = f"model"
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    images_dir = model_dir + "/images"
    if not os.path.exists(images_dir):
        os.mkdir(images_dir)
    return model_dir, images_dir



def importance_plot(xgb_regressor,images_dir):
    # Plotting feature importances using the plot_importance function from XGBoost
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

    # ---- AXIS FORMATTING (THIS IS THE KEY) ----
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))   # tick every 3 days
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


# Plotting functionality INSERT HERE
def save_model(model, model_dir):
    # Saving the XGBoost regressor object as a json file in the model directory
    booster = model.get_booster()
    booster.save_model(model_dir + "/model.json")
    #model.save_model(model_dir + "/model.json")

def saving_model_hops( fv, mse, r2, project, dir):
    model_reg = project.get_model_registry()

    model_elprice = model_reg.python.create_model(
        name="model_elprice",
        metrics= {"MSE": str(mse), "R2": str(r2)},
        feature_view = fv,
        description = "Predictions of electricity prices for Sweden Area SE3."
    )
    model_elprice.save(dir)

if mode_select == 3:
    project = connect_to_hopsworks_project()

    fs = create_fs(project)

    model_dir, images_dir = create_dirs()


    price_fg, weather_1_fg, weather_2_fg = get_fgs(fs, "stockholm", "goteborg")

    selected_f = set_selected_features(price_fg,weather_1_fg,weather_2_fg)

    fv = create_fv(fs, selected_f)

    X_tr, X_ts, y_tr, y_ts = set_training_data(fv)

    X_tr_f, X_ts_f =get_features(X_tr, X_ts)


    model = model_creation(X_tr_f,y_tr)

    y_pr, mse, r2 = pred_eval(model,X_ts_f,y_ts)

    plot_df = df_creation(y_ts,y_pr,X_ts)

    save_model(model, model_dir)

    saving_model_hops(fv, mse,r2,project,model_dir)

    importance_plot(model,images_dir)
    plot_train("SE3",plot_df,images_dir,True)


###### PART 4 INFERENCE ######

def get_model(project):
    mr = project.get_model_registry()
    
    retrieved_model = mr.get_model(
        name=f"model_elprice",
        version=1,
    )
    
    fv = retrieved_model.get_feature_view()
    
    saved_model_dir = retrieved_model.download()

    retrieved_xgboost_model = XGBRegressor()

    retrieved_xgboost_model.load_model(saved_model_dir + f"/model.json")

    return retrieved_xgboost_model, fv

def get_forecast_weather(fs,today,city1, city2):
    # join the data for the two feature groups into batch data on date 
    weather_fg_1 = fs.get_feature_group(
        name=f"weather_{city1.lower()}",
        version=1,
    )
    batch_data_1 = weather_fg_1.filter(weather_fg_1.date >= today).read()

    weather_fg_2 = fs.get_feature_group(
        name=f"weather_{city2.lower()}",
        version=1,
    )
    batch_data_2 = weather_fg_2.filter(weather_fg_2.date >= today).read()

    combined_data = pd.merge(batch_data_1, batch_data_2, on="date", how="inner")

    combined_data = combined_data.sort_values("date").reset_index(drop=True)

    return combined_data

def fill_features(hour,batch_data,prev_data):
    prev_data["date"] = pd.to_datetime(prev_data["date"], utc=True)
    prev_data = prev_data.sort_values("date").reset_index(drop=True)
    price_series = prev_data["price_sek_per_kwh"]
    
    batch_data.loc[batch_data["date"] == hour,"hour"] = hour.hour
    batch_data.loc[batch_data["date"] == hour,"day"] = hour.day
    batch_data.loc[batch_data["date"] == hour,"month"] = hour.month
    #batch_data.loc[batch_data["date"] == hour,"price_1h_ago"]  = price_series.iloc[-1]
    #batch_data.loc[batch_data["date"] == hour,"price_2h_ago"]  = price_series.iloc[-2]
    #batch_data.loc[batch_data["date"] == hour,"price_3h_ago"]  = price_series.iloc[-3]
    batch_data.loc[batch_data["date"] == hour,"price_24h_ago"] = price_series.iloc[-24]
    #batch_data.loc[batch_data["date"] == hour,"price_48h_ago"] = price_series.iloc[-48]
    #batch_data.loc[batch_data["date"] == hour,"price_72h_ago"] = price_series.iloc[-72]
    

    
    
    #current_hour = hour.hour
    

    # Filter for the same hour across all days in the history
    #same_hour_data = prev_data[prev_data["date"].dt.hour == current_hour]
    
    
    
    
    #batch_data.loc[batch_data["date"] == hour, "price_avg_same_hour_last_week"] = same_hour_data["price_sek_per_kwh"].rolling(7).mean().iloc[-1]
    
    
    
    #batch_data.loc[batch_data["date"] == hour, "price_avg_same_hour_last_month"] = same_hour_data["price_sek_per_kwh"].rolling(30).mean().iloc[-1]

    # Feature: Rolling 168h average (simple contiguous rolling)
    #batch_data.loc[batch_data["date"] == hour, "price_last_168h_avg"] = price_series.rolling(168).mean().iloc[-1]
    
    
    
    return batch_data

def add_first_price_features(fs,batch_data):
    import pytz
    
    first_hour = (datetime.datetime.now(tz=pytz.UTC)).replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=0) # REMOVE
    
    price_group = fs.get_feature_group(name=f"price_swedene3", version=1)
    first_init_roll = price_group.filter(price_group.date >= first_hour -  timedelta(31)).read()
    
    #print(first_init_roll)
    first_init_roll = first_init_roll.sort_values("date").reset_index(drop=True)
    batch_data = fill_features(first_hour,batch_data,first_init_roll)
    

    return batch_data, first_hour, first_init_roll


def predictions(hour,batch_data,retrieved_xgboost_model,first_init_roll):
    

    batch_data["date"] = pd.to_datetime(batch_data["date"], utc=True)
    time_obj = pd.to_datetime(hour, utc=True)
    #with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #    print(batch_data)
    
    
    for i in range(len(batch_data)):
        
    
        mask = batch_data["date"] == time_obj
        
        if not mask.any():
            break
        
        """X_old_copy = batch_data.loc[mask,["month", 
        "price_1h_ago", "price_2h_ago", "price_3h_ago", 
        "price_24h_ago", "price_48h_ago", "price_72h_ago",
        "price_avg_same_hour_last_week", 
        "price_avg_same_hour_last_month",
        "price_last_168h_avg", "temperature_2m_stockholm", "precipitation_stockholm", 
                                 "wind_speed_10m_stockholm", "cloud_cover_stockholm", "temperature_2m_goteborg", "precipitation_goteborg", 
                                 "wind_speed_10m_goteborg", "cloud_cover_goteborg"]]"""
        
        X = batch_data.loc[mask,["hour", "day", "month",  
        "price_24h_ago", "temperature_2m_stockholm", "precipitation_stockholm", 
                                 "wind_speed_10m_stockholm", "cloud_cover_stockholm", "temperature_2m_goteborg", "precipitation_goteborg", 
                                 "wind_speed_10m_goteborg", "cloud_cover_goteborg"]]
    
        y_pred = retrieved_xgboost_model.predict(X)[0]
        
        batch_data.loc[mask, "pred_price"] = y_pred
        
        new_data = {
            "date": time_obj,
            "price_sek_per_kwh": y_pred,
        }
        
        
    
        first_init_roll = pd.concat([first_init_roll, pd.DataFrame([new_data])], ignore_index=True)
        first_init_roll = first_init_roll.sort_values("date").reset_index(drop=True)
        time_next = time_obj + timedelta(hours=1)
    
        if(batch_data["date"]== time_next).any():
            batch_data = fill_features(time_next,batch_data,first_init_roll)
            # add latest to first init
        time_obj = time_next
        
    batch_data['days_before_forecast_day'] = (np.arange(len(batch_data)) // 24 + 1).astype(np.int32)
    
    
    return batch_data

import matplotlib.dates as mdates

def plot_price_forecast(region: str, df: pd.DataFrame, file_path: str, hindcast=False):
    fig, ax = plt.subplots(figsize=(14, 6))

    df = df.copy()
    # Ensure date is datetime objects for matplotlib
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    # Plot Predicted Price
    ax.plot(
        df["date"],
        df["pred_price"],
        label="Predicted Electricity Price",
        color="red",
        linewidth=1.8
    )

    # Plot Actual Price if requested
    if hindcast and "price_sek_per_kwh" in df.columns:
        ax.plot(
            df["date"],
            df["price_sek_per_kwh"],
            label="Actual Price",
            color="black",
            linewidth=1.8,
            alpha=0.8
        )

    # ---- UPDATED AXIS FORMATTING FOR HOURLY VIEW ----
    # Major ticks every 12 hours to keep the labels readable
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))   
    
    # Format showing Year-Month-Day Hour:Minute
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))

    # Minor ticks every 3 hours (ticks without labels) to show granularity
    ax.xaxis.set_minor_locator(mdates.HourLocator(interval=3))

    ax.set_xlabel("Time (Hourly)")
    ax.set_ylabel("SEK / kwh")
    if hindcast:
        ax.set_title(f"Hourly Electricity Price Hindcast {region}")
    else:
        ax.set_title(f"Hourly Electricity Price Forecast {region}")

    # Rotate labels so they don't overlap
    plt.xticks(rotation=45)
    ax.grid(True, which='both', alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(file_path, dpi=300)
    plt.close()

    return file_path

def monitoring_fg(batch_data,fs):
    # Get or create feature group
    monitor_fg = fs.get_or_create_feature_group(
        name=f"price_prediction_monitoring",
        description='Electricity price real vs predicted comparison',
        version=1,
        primary_key=['date','days_before_forecast_day'],
        event_time="date"
    )
    print(batch_data["days_before_forecast_day"], batch_data["days_before_forecast_day"].dtype)
    monitor_fg.insert(batch_data, wait=True)

    return monitor_fg

def monitor_df(monitor_fg):
    
    monitoring_df = monitor_fg.filter(monitor_fg.days_before_forecast_day == 1).read()
    #print("monitordf")
    #print(monitoring_df)

    return monitoring_df

def elprice_pred_df(fs):
    electricity_fg = fs.get_feature_group(name="price_swedene3", version=1)
    electricity_df = electricity_fg.read()

    return electricity_fg, electricity_df

def hindcast(electricity_price_df, monitoring_df):
    outcome_df = electricity_price_df[['date', 'price_sek_per_kwh']].sort_values(by=['date'])
    preds_df =  monitoring_df[['date', 'pred_price']].sort_values(by=['date'])
    #print(outcome_df)
    #print(preds_df)
    
    hindcast_df = pd.merge(preds_df, outcome_df, on="date")
    hindcast_df = hindcast_df.sort_values(by=['date'])
    #print(hindcast_df)
    return hindcast_df



def upload_to_hops(project, today,pred_path,hind_path):
    dataset_api = project.get_dataset_api()
    str_today = today.strftime("%Y-%m-%d")
    if dataset_api.exists("Resources/SE3") == False:
        dataset_api.mkdir("Resources/SE3")
    dataset_api.upload(pred_path, f"Resources/SE3/forecast_{str_today}", overwrite=True)
    dataset_api.upload(hind_path, f"Resources/SE3/hindcast_{str_today}", overwrite=True)
    
    
    

if mode_select == 4:

    cities = {"stockholm":[59.3294,18.0687], "goteborg":[57.7072,11.9668]}
    keys = list(cities.keys())
    
    project = connect_to_hopsworks_project()

    fs = create_fs(project)

    model, fv = get_model(project)
    starttime=first_hour = (
        datetime.datetime.now(pytz.timezone("Europe/Stockholm"))
        .replace(hour=0, minute=0, second=0, microsecond=0)
        - timedelta(days=1) 
    ).astimezone(pytz.UTC)
    combined_data = get_forecast_weather(fs, starttime, keys[0], keys[1])

    batch_data, first_hour, first_roll = add_first_price_features(fs, combined_data)
    first_roll = first_roll.sort_values("date").reset_index(drop=True)
    pred_data = predictions(first_hour, batch_data, model, first_roll)
    
    model_dir,images_dir = create_dirs()
    #with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #    print(batch_data)
    forecast_path = plot_price_forecast("SE3",batch_data, images_dir + "/forecast.png", False)

    monitor_fg = monitoring_fg(batch_data,fs)

    monitoring_df = monitor_df(monitor_fg)

    elec_fg, elec_df = elprice_pred_df(fs)

    hindcast_df = hindcast(elec_df,monitoring_df)

    hindcast_path = plot_price_forecast("SE3",hindcast_df, images_dir + "/hindcast.png", True)
    today = datetime.datetime.now()
    upload_to_hops(project,today, forecast_path,hindcast_path)
    











