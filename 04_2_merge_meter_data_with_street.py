import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from chinese_calendar import is_workday, is_holiday
from rdd import rdd
import datetime
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter
import statsmodels.api as sm



def pre_process_street(street_name_area):
    street_name_area = street_name_area.rename(columns = {'before_are':'before_area'})
    # fix dongge St due to identification error in GIS
    street_name_area.loc[street_name_area['Name_CHN']=='东葛路','after_area'] = 'A-after'
    street_name_area['before_area'] = street_name_area[['before_area']].fillna('C-before')
    street_name_area['after_area'] = street_name_area[['after_area']].fillna('C-after')
    street_name_area = street_name_area.drop_duplicates(['Name_CHN','Name_ENG','before_area','after_area'])
    #### fix bugs of some roads
    road_id = [113,145,291]
    street_name_area.loc[street_name_area['id'].isin(road_id),'before_area'] = 'B-before'
    #check
    street_name_area_group = street_name_area.groupby(['before_area','after_area'])['id'].count().reset_index()

    street_name_area.to_csv('data/street_name_area2.csv', encoding = 'utf_8_sig', index = False)





def pre_process_meter_data(meter_data,street_name_area, street_data_time):

    # only need road with full data
    street_data_time['Min_data_time'] = pd.to_datetime(street_data_time['Min_data_time'],
                                                            format = '%Y-%m-%d %H:%M:%S')
    street_data_time['Max_data_time'] = pd.to_datetime(street_data_time['Max_data_time'],
                                                          format='%Y-%m-%d %H:%M:%S')
    #type(street_data_time['Min_data_time'].iloc[0])

    useful_road = street_data_time.loc[(street_data_time['Min_data_time']<pd.Timestamp(2014, 9, 2))&
                                       (street_data_time['Max_data_time']>pd.Timestamp(2015, 8, 30))]

    useful_road = useful_road.merge(street_name_area,left_on =['Street'],right_on = ['Name_CHN'],how = 'left')
    #useful_road.to_csv('data/useful_road.csv', encoding='utf_8_sig',index=False)
    print('Total data road', len(useful_road))
    print('Useful road no data', len(useful_road.loc[useful_road['Name_CHN'].isna()]))

    # count different category
    useful_road = useful_road.dropna()
    useful_road_cate = useful_road.groupby(['before_area','after_area'])['Street'].count()
    useful_road_cate = useful_road_cate.reset_index(drop=False)
    print(useful_road_cate)
    meter_data_info = meter_data.merge(useful_road[['Street','before_area','after_area','id']],on=['Street'])
    meter_data_info = meter_data_info.drop(columns = ['Street'])
    # meter_data_info = meter_data_info.drop_duplicates()
    # meter_data_info.to_csv('data/meter_data_info.csv',index=False)
    ######
    num_data_in_meter = meter_data_info.groupby(['before_area','after_area'])['id'].count()
    num_data_in_meter = num_data_in_meter.reset_index(drop=False)
    print(num_data_in_meter)
    num_data_in_meter_before = num_data_in_meter.groupby(['before_area'])['id'].sum()
    num_data_in_meter_before = num_data_in_meter_before.reset_index(drop=False)
    print(num_data_in_meter_before)
    num_data_in_meter_after = num_data_in_meter.groupby(['after_area'])['id'].sum()
    num_data_in_meter_after = num_data_in_meter_after.reset_index(drop=False)
    print(num_data_in_meter_after)
    print('total',sum(num_data_in_meter_after['id']))





if __name__ == '__main__':


    #############Process st data
    St_parking_area = pd.read_csv('data/St_parking_area.csv')
    pre_process_street(St_parking_area)
    # # #############Process meter data
    street_name_area = pd.read_csv('data/street_name_area2.csv')
    meter_data = pd.read_csv('data/meter_data.csv')
    street_data_time = pd.read_csv('data/all_street_in_meter.csv')
    pre_process_meter_data(meter_data, street_name_area, street_data_time)