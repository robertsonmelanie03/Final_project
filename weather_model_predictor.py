from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv('homework/datasets/weather_prediction_dataset.csv')

def fit_model(data):

    #Will pass in the data
    #This creates the "previous-step" features
    data["prev_temp"] = data["DE_BILT_temp_mean"].shift(1)
    data["prev_humidity"] = data["DE_BILT_humidity"].shift(1)
    data["prev_precip"] = data["DE_BILT_precipitation"].shift(1)
    data["prev_wind"] = data["DE_BILT_wind_speed"].shift(1)  
    data["prev_pressure"] = data["DE_BILT_pressure"].shift(1)
    data["prev_cloud_cover"] = data["DE_BILT_cloud_cover"].shift(1)
    data["prev_radiation"] = data["DE_BILT_global_radiation"].shift(1)
    data["month"] = data["MONTH"].shift(1)

    #Drposs missing row after the row shift
    data = data.dropna()     
    feature_cols = ['prev_temp', 'prev_humidity', 'prev_precip', 'prev_wind', 'prev_pressure', 'prev_cloud_cover', 'prev_radiation', 'month']
    targets = ['DE_BILT_temp_mean', 'DE_BILT_humidity', 'DE_BILT_precipitation', 'DE_BILT_wind_speed']

    X = data[feature_cols]
    Y = data[targets]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.2, random_state= 42, shuffle= False)

    model = RandomForestRegressor()

    model.fit(X_train, Y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(Y_test, y_pred)
    r2 = r2_score(Y_test, y_pred)

    print("MSE:", mse)
    print("R^2:", r2)

    # Compare actual vs predicted
    pred_df = pd.DataFrame(y_pred, columns= targets, index= Y_test.index)

    comparison = pd.concat([Y_test.reset_index(drop=True), pred_df.reset_index(drop=True)], axis=1, keys=["Actual", "Predicted"])

    print(comparison.head(10))

    return model

def predict_next_weather(model, current_weather, context):

    df = pd.DataFrame([{
    "prev_temp": current_weather["temp"], 
    "prev_humidity": current_weather["humidity"],
    "prev_precip": current_weather["precip"],
    "prev_wind": current_weather["wind"],
    "prev_pressure": context["pressure"],
    "prev_cloud_cover": context["cloud_cover"],
    "prev_radiation": context["radiation"],
    "month": context["month"]}])

    #Have to do this because df returns an array with double brackets, so liek a list of a list
    #so without it, it'd return like df[0] = [7.31, 0.9, 0.1, 1.4] df[1] = error
    prediction = model.predict(df)[0]

    return {'temp': prediction[0], 'humidity': prediction[1], 'precip': prediction[2], 'wind': prediction[3]}

row = data.iloc[0]

current_weather = {
    "temp": row["DE_BILT_temp_mean"], 
    "humidity": row["DE_BILT_humidity"],
    "precip": row["DE_BILT_precipitation"],
    "wind": row["DE_BILT_wind_speed"],}

context = {
    "pressure": row["DE_BILT_pressure"],
    "cloud_cover": row["DE_BILT_cloud_cover"],
    "radiation": row["DE_BILT_global_radiation"],
    "month": row["MONTH"]
}

model = fit_model(data)

for step in range(100):

    current_weather = predict_next_weather(model, current_weather, context)
    print(current_weather)
