# -*- coding: utf-8 -*-
import pandas as pd

raw_2014 = pd.read_csv('data/Raw_2014.csv')
raw_2015 = pd.read_csv('data/Raw_2015.csv')
print (len(raw_2014))
print (len(raw_2015))
weather = pd.read_csv('data/weather_survey_month_Nanning.csv')
raw_2014 = raw_2014.rename(columns={'时间':'date'})
raw_2014['date'] = pd.to_datetime(raw_2014['date'].apply(str))
raw_2015 = raw_2015.rename(columns={'时间':'date'})
raw_2015['date'] = pd.to_datetime(raw_2015['date'].apply(str))
weather['ymd'] = pd.to_datetime(weather['ymd'].apply(str))
weather['wind'] = weather['fengli'].apply(lambda x: 0.5*(float(x.split('~')[0])+float(x.split('~')[1])) if len(
    (x.split('~')))>1 else float(x))
weather['sun'] = weather['tianqi'].apply(lambda x: 1 if 'sun' in x else 0)
weather['cloud'] = weather['tianqi'].apply(lambda x: 1 if 'cloud' in x else 0)
weather['rain'] = weather['tianqi'].apply(lambda x: 1 if 'rain' in x else 0)
weather['temp'] = (weather['Min_Temp'] + weather['Max_Temp'])/2
#print (len(raw_2014))
raw_2014 = raw_2014.merge(weather[['ymd','sun','cloud','rain','temp','wind']],left_on=['date'], right_on = ['ymd'],sort=False,how='left')
#print (len(raw_2014))

#print (len(raw_2015))
raw_2015 = raw_2015.merge(weather[['ymd','sun','cloud','rain','temp','wind']],left_on=['date'], right_on = ['ymd'],sort=False,how='left')
#print (len(raw_2015))

#--------------------day of week
raw_2014['day_week'] = raw_2014['date'].apply(lambda x: x.weekday())
raw_2014['day_week'] +=1 
#  Monday is 1 and Sunday is 7 
raw_2015['day_week'] = raw_2015['date'].apply(lambda x: x.weekday())
raw_2015['day_week'] +=1 
raw_2014.to_csv('data/Raw_2014_new.csv',index=False,encoding="utf-8-sig")
raw_2015.to_csv('data/Raw_2015_new.csv',index=False,encoding="utf-8-sig")
