import pandas as pd
import numpy as np
import time

def combine_all(file_list):
    data = pd.DataFrame()
    for file in file_list:
        data_temp = pd.read_excel(file, sheet_name=0, header=0)
        data = pd.concat([data, data_temp],sort=False)
        print('current file', file)
    data = data.rename(columns = {'街道':'Street','入位时间':'Parking_start_time',
                                  '出车时间':'Parking_end_time','停车时长（分钟）':'Parking_duration'})
    #
    data['Street'] = data['Street'].str.replace(' ', '')
    tic = time.time()
    data['PS_year'] = data['Parking_start_time'].dt.year
    data['PS_month'] = data['Parking_start_time'].dt.month
    data['PS_day'] = data['Parking_start_time'].dt.day
    data['PS_hour'] = data['Parking_start_time'].dt.hour
    data['PS_min'] = data['Parking_start_time'].dt.minute
    data['PS_sec'] = data['Parking_start_time'].dt.second

    data['PE_year'] = data['Parking_end_time'].dt.year
    data['PE_month'] = data['Parking_end_time'].dt.month
    data['PE_day'] = data['Parking_end_time'].dt.day
    data['PE_hour'] = data['Parking_end_time'].dt.hour
    data['PE_min'] = data['Parking_end_time'].dt.minute
    data['PE_sec'] = data['Parking_end_time'].dt.second

    data['day_of_week'] = data['Parking_start_time'].dt.weekday
    # The day of the week with Monday=0, Sunday=6.

    # data['Parking_start_time'] = pd.to_datetime(data['Parking_start_time'], format ='%Y-%m-%d %H:%M:%S')
    # data['Parking_end_time'] = pd.to_datetime(data['Parking_end_time'], format='%Y-%m-%d %H:%M:%S')
    print('process datetime time', time.time() - tic)
    # afc['txn_timestamp'] = afc['transaction_dtm'].apply(
    #     lambda x: int(x.split(' ')[1].split(':')[0]) * 3600 +
    #               int(x.split(' ')[1].split(':')[1]) * 60 + int(x.split(' ')[1].split(':')[2]))

    # get street data time
    street_list = data.groupby(['Street'])['Parking_start_time'].agg(['min','max'])
    street_list = street_list.reset_index(drop=False)
    street_list = street_list.rename(columns = {'min':'Min_data_time','max':'Max_data_time'})
    street_list.to_csv('data/all_street_in_meter.csv',encoding='utf_8_sig',index=False)

    data = data.drop(columns=['Parking_start_time','Parking_end_time'])
    data.to_csv('data/meter_data.csv', encoding='utf_8_sig',index=False)





if __name__ == '__main__':
    file_path = '../Meter_data/'
    month_list = ['201409','201410','201411','201412','201501','201502',
                  '201503','201504','201505','201506','201507','201508'] #201509 data may have some problems, drop it, too much parking volume
    # month_list = ['201409']
    file_list = [file_path + month + '停车数据清单' + '.xlsx' for month in month_list]

    combine_all(file_list)