#!/usr/bin/env python
# coding: utf-8

import xgboost as xgb
import xarray as xr
import pandas as pd
import numpy as np
import gc
from flaml import AutoML
import shap
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import sys
import ast
import logging 
import pickle

"""
# how to use it:
e.g.,
$ python train_all_gridcell.py "daily" "["aod","emission","met","gas"]"

"""

time_scale = sys.argv[1] 
input_feature = sys.argv[2]
feature_cat = list(map(str,input_feature.strip('[]').split(','))) # process from input to list
feature_join = "_".join(feature_cat) # for file name

label = "PM25"
# ================================ #
if time_scale == "daily":
    print("start daily data")
    df_train = pd.read_parquet('your_data/c_r_daily_train.gzip')
    df_test = pd.read_parquet('your_data/c_r_daily_test.gzip')
    feature_dict = {
    "gas":['CO_trop', 'SO2_trop', 'NO2_trop', 'CH2O_trop', 'NH3_trop'],
    "aod":['AOT_C', 'AOT_DUST_C'],
    "met":['T2M', 'PBLH', 'U10M', 'V10M', 'PRECTOT', 'RH'],
    "emission":['EmisDST_Natural', 
                'EmisNO_Fert', 'EmisNO_Lightning', 'EmisNO_Ship', 'EmisNO_Soil']}
    # ref https://stackoverflow.com/questions/952914/how-to-make-a-flat-list-out-of-a-list-of-lists
    feature_ls = sum([feature_dict[k] for k in feature_cat],[])
    print(feature_ls)

elif time_scale == "monthly_le":
    print("start monthly_le data")
    df_train = pd.read_parquet('your_data/c_r_monthly_train.gzip')
    df_test = pd.read_parquet('your_data/c_r_monthly_test.gzip')
    feature_dict = {
    "pm":["PM25"],
    "gas":['CO_trop', 'SO2_trop', 'NO2_trop', 'CH2O_trop', 'NH3_trop'],
    "aod":['AOT_C', 'AOT_DUST_C'],
    "met":['T2M', 'PBLH', 'U10M', 'V10M', 'PRECTOT', 'RH'],
    "emission":['EmisDST_Natural', 
                'EmisNO_Fert', 'EmisNO_Lightning', 'EmisNO_Ship', 'EmisNO_Soil']}
    feature_ls = sum([feature_dict[k] for k in feature_cat],[])
    print(feature_ls)

elif time_scale == "monthly":
    print("start monthly data")
    df_train = pd.read_parquet('your_data/c_r_monthly_train.gzip')
    df_test = pd.read_parquet('your_data/c_r_monthly_test.gzip')
    feature_dict = {
    "pm":["PM25"],
    "gas":['CO_trop', 'SO2_trop', 'NO2_trop', 'CH2O_trop', 'NH3_trop'],
    "aod":['AOT_C', 'AOT_DUST_C'],
    "met":['T2M', 'PBLH', 'U10M', 'V10M', 'PRECTOT', 'RH'],
    "emission":['EmisDST_Natural', 
                'EmisNO_Fert', 'EmisNO_Lightning', 'EmisNO_Ship', 'EmisNO_Soil',
                'EmisBC_Anthro', 'EmisBC_BioBurn', 
                'EmisCH2O_Anthro', 'EmisCH2O_BioBurn', 
                'EmisCO_Anthro', 'EmisCO_BioBurn', 'EmisCO_Ship', 
                'EmisNH3_Anthro', 'EmisNH3_BioBurn', 'EmisNH3_Natural', 
                'EmisNO_Aircraft', 'EmisNO_Anthro', 'EmisNO_BioBurn', 
                'EmisOC_Anthro', 'EmisOC_BioBurn',  
                'EmisSO2_Aircraft', 'EmisSO2_Anthro', 'EmisSO2_BioBurn',
                'EmisSO4_Anthro']}
    feature_ls = sum([feature_dict[k] for k in feature_cat],[])
    print(feature_ls)

else:
    print("Exit! Please type 'daily' or 'monthly'")
    sys.exit()

def workflow(feature_ls, label, df_train, df_test):
    X_train = df_train[feature_ls].copy()
    y_train = df_train[label].copy()
    X_test = df_test[feature_ls].copy()
    y_test = df_test[label].copy()

    automl = AutoML()
    automl_settings = {
        "metric": 'r2',
        "estimator_list": 'auto',
        "task": 'regression',
        "log_file_name": "./"+time_scale+"_"+feature_join+"_"+loc+".log"
    }
    
    automl.fit(X_train=X_train, y_train=y_train,
               verbose=0, time_budget=5400, seed=66,
               **automl_settings)
    print(f"best_etimator:{automl.best_estimator}")


    # print training error
    y_pred = automl.predict(X_train)
    print("train r2_score:",
        r2_score(y_true = y_train, y_pred = y_pred))
    print("train root mean_squared_error:",
        mean_squared_error(y_true = y_train, y_pred = y_pred, squared = False))
    print("train mean_absolute_error:",
        mean_absolute_error(y_true = y_train, y_pred = y_pred))
    del y_pred
    gc.collect()

    
    # print testing error
    y_pred = automl.predict(X_test)
    print("test r2_score:",
        r2_score(y_true = y_test, y_pred = y_pred))
    print("test root mean_squared_error:",
        mean_squared_error(y_true = y_test, y_pred = y_pred, squared = False))
    print("test mean_absolute_error:",
        mean_absolute_error(y_true = y_test, y_pred = y_pred))
    
    #return automl.model.estimator, automl.best_config
    return automl
# ================================ #

for loc in ["E","S","W","N"]:
    print("start",loc)
    df_train_s = df_train[df_train["region"]==loc]
    df_test_s = df_test[df_test["region"]==loc]
    automl = workflow(feature_ls, label, df_train_s, df_test_s)
    print("save at: ./"+time_scale+"_"+feature_join+"_"+loc+".pkl")
    # save model
    with open("./"+time_scale+"_"+feature_join+"_"+loc+".pkl", 'wb') as f:
        pickle.dump(automl, f, pickle.HIGHEST_PROTOCOL)

    del df_train_s, df_test_s, automl
    gc.collect()
    print("\n")



