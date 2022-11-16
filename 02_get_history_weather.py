# -*- coding: utf-8 -*-
import re
import requests

import json

import pandas as pd

def getWeather(year_month):
    #
    url='https://tianqi.2345.com/t/wea_history/js/'+year_month+'.js'
    headers={'User-Agent':'Mozilla/5.0 (Windows; U; Windows NT 6.1; en-US; rv:1.9.1.6) Gecko/20091201 Firefox/3.5.6','referer':'link'}
    str_content = requests.get(url,headers=headers).content.decode('gbk')
    #print (type(str_content))
    str_content=str_content.replace('var weather_str=','')
    str_content=str_content.replace(';','')
    str_content=str_content.replace('香港','HongKong')
    str_content=str_content.replace('℃','')
    str_content=str_content.replace("'", '"')
    str_content=str_content.replace("aqiInfo", '"aqiInfo"')
    str_content=str_content.replace("aqiLevel", '"aqiLevel"')
    str_content=str_content.replace("maxAqi", '"maxAqi"')
    str_content=str_content.replace("minAqi", '"minAqi"')
    str_content=str_content.replace("avgAqi", '"avgAqi"') 
    str_content=str_content.replace('"maxAqi"Info', '"maxAqiInfo"')
    str_content=str_content.replace('"maxAqi"Date', '"maxAqiDate"')
    str_content=str_content.replace('"maxAqi"Level', '"maxAqiLevel"')
    str_content=str_content.replace('"minAqi"Info', '"minAqiInfo"')
    str_content=str_content.replace('"minAqi"Date', '"minAqiDate"')
    str_content=str_content.replace('"minAqi"Level', '"minAqiLevel"')
    str_content=str_content.replace("aqi:", '"aqi":')
    str_content=str_content.replace("city", '"city"')
    str_content=str_content.replace("tqInfo", '"tqInfo"')
    str_content=str_content.replace("ymd", '"ymd"')
    str_content=str_content.replace("bWendu", '"bWendu"')
    str_content=str_content.replace("yWendu", '"yWendu"')
    str_content=str_content.replace("tianqi", '"tianqi"')
    str_content=str_content.replace("fengxiang", '"fengxiang"')
    str_content=str_content.replace("fengli", '"fengli"')
    str_content=str_content.replace("maxWendu", '"maxWendu"')
    str_content=str_content.replace("minWendu", '"minWendu"')
    str_content=str_content.replace("avg\"bWendu\"", '"avgbWendu"')
    str_content=str_content.replace("avg\"yWendu\"", '"avgyWendu"')
    str_content=str_content.replace("无持续风向","no wind direction")
    str_content=str_content.replace("小到中雨","rain")
    str_content=str_content.replace("大到暴雨","heavy_rain")
    str_content=str_content.replace("中到大雨","heavy_rain")
    str_content=str_content.replace("大暴雨","heavy_rain")
    str_content=str_content.replace("雷阵雨","rain")
    str_content=str_content.replace("东北风","NorthEast")
    str_content=str_content.replace("西南风","SouthWest")
    str_content=str_content.replace("东南风","SouthEast")
    str_content=str_content.replace("西北风","NorthWest")
    str_content=str_content.replace("微风","1.5")
    str_content=str_content.replace("多云","cloud")
    str_content=str_content.replace("3-4级","3.5")
    str_content=str_content.replace("4-5级","4.5")
    str_content=str_content.replace("6-7级","6.5")
    str_content=str_content.replace("5-6级","5.5")
    str_content=str_content.replace("11-12级","11.5")
    str_content=str_content.replace("雷雨","heavy_rain")
    str_content=str_content.replace("阵雨","rain")
    str_content=str_content.replace("小雨","rain")
    str_content=str_content.replace("暴雨","heavy_rain")
    str_content=str_content.replace("大雨","heavy_rain")
    str_content=str_content.replace("中雨","rain")
    str_content=str_content.replace("南风","South")
    str_content=str_content.replace("东风","East")
    str_content=str_content.replace("西风","West")
    str_content=str_content.replace("晴","sun")
    str_content=str_content.replace("北风","North")
    str_content=str_content.replace("阴","cloud")
    str_content=str_content.replace("雾","cloud")	
    #print (str_content)
    dict_content=json.loads(str_content)
    #print (type(dict_content))
    df=pd.DataFrame(dict_content['tqInfo'])
    df=df.dropna(axis=0,how='all')  
    cols = df.columns.tolist()
    #print (cols)
    if int(year_month.split('_')[1][0:4])<2016:
        cols = [cols[5],cols[3],cols[0],cols[4],cols[2],cols[1]]
    else:
        cols = [cols[8],cols[6],cols[3],cols[7],cols[5],cols[4]]
    df=df[cols]
    df.columns=['ymd', 'tianqi', 'Max_Temp', 'Min_Temp', 'fengxiang', 'fengli']
    #print (cols)
    df=df.sort_values(by='ymd')
    #print (df)
    return df

if __name__ == "__main__":
    
    year_month_list=['59431_20149','59431_20154','59431_20155']
    #year_month_list=['59431_20161']
    count=0
    for year_month in year_month_list:
        if count==0:
            df_final=getWeather(year_month)
            count+=1
        else:
            df_final=pd.concat([df_final,getWeather(year_month)])
    df_final.to_csv('data/weather_survey_month_Nanning.csv',index=False)    
            