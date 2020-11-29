import eazyml
import pandas as pd
import numpy as np

def main_function(county_name):
    big_df = pd.read_csv("webserver\\big_data.csv")
    dates = big_df["Date"]
    county_column = big_df[county_name]
    
    average = sum(list(big_df[county_name]))/310

    result = pd.concat([dates, county_column], axis=1)
    result.to_csv("webserver\\dataset.csv", index=False)

    #authentication to use api
    username = 'adsarwaikar@ctemc.org'
    password = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJqdGkiOiJkMzk0YzBkMy0yYmMxLTRkNDYtOWRiMS01NjZhYWUwZmMzOWUiLCJleHAiOjE2MDU0Njg4NTMsImZyZXNoIjpmYWxzZSwiaWF0IjoxNjA1MzgyNDUzLCJ0eXBlIjoiYWNjZXNzIiwibmJmIjoxNjA1MzgyNDUzLCJpZGVudGl0eSI6IkFkaXR5YSBTYXJ3YWlrYXIifQ._sIXpWubJ5DdGuhECUsgLNI2tXai9Ih3BsVxOJIFo-s'
    train_file_path = 'webserver\\dataset.csv'
    
    resp = eazyml.ez_auth(username, None, password)
    auth_token = resp["token"]
    
    options = {
        "id": "null",
        "impute": "yes",
        "outlier": "yes",
        "discard": "null",
        "accelerate": "yes",
        "shuffle": "no",
        "outcome" : county_name
    }
    
    ez_model_config = {
        "model_type" : "timeseries",
        "derive_text" : "no",
        "derive_numeric": "no",
        "accelerate": "yes",
        "date_time_column": "Date"
    }
    
    #loading the training data
    resp = eazyml.ez_load(auth_token, train_file_path, options)
    print(resp)
    dataset_id = resp["dataset_id"]
    
    #building the model
    resp = eazyml.ez_init_model(auth_token, dataset_id, ez_model_config)
    resp = eazyml.ez_get_models(auth_token)
    print(resp)
    model_id = resp["dataframe"]["data"][0][4] 
    
    prediction_df = pd.DataFrame({'Date':["6/30/2020", "12/31/2020"], 'Monmouth':["",""]})
    prediction_df.to_csv("webserver\\prediction.csv", index=False)
    #getting final response with answers and displaying to user
    response = eazyml.ez_predict(auth_token, model_id, 'webserver\\prediction.csv')
    half_year = float(response["predictions"]["data"][0][2])
    if half_year > average:
        half_year_statement = True
    else:
        half_year_statement = False
    full_year = float(response["predictions"]["data"][1][2])
    if full_year > average:
        full_year_statement = True
    else:
        full_year_statement = False
    return half_year, half_year_statement, full_year, full_year_statement