# Electricity Price Prediction in Sweden SE3 Area Dashboard

![Hopsworks Logo](../titanic/assets/img/logo.png)

{% include price_predict.html %}

{% assign names = "anan_nakagawacho_kuroji,jinryo_main_street,kitajima_tainohama,mima_wakimachi_oaza_inoshiri,miyoshi_ikedacho_machi,naruto_muyacho_tateiwa,nishinoji_main_street,tokushima_shinkuracho" | split: ',' %}

{% for name in names %}
### {{ name | capitalize }} Air Quality

![Forecast](./assets/img/pm25_forecast_{{ name }}.png)

# Model Performance Monitoring

1-Day Hindcast: Predictions vs Outcomes

![Hindcast](./assets/img/pm25_hindcast_1day_{{ name }}.png)

{% endfor %}
