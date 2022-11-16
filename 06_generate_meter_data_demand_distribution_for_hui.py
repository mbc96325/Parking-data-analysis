import pandas as pd
import numpy as np

def generate_area_volume_distribution(meter_data_info,st_data):
    meter_data_info['date_time'] = meter_data_info['PS_year'].astype('str') + '/' + \
                                   meter_data_info['PS_month'].astype('str') + '/' + meter_data_info['PS_day'].astype('str')

    meter_data_info['date_time'] = pd.to_datetime(meter_data_info['date_time'],format='%Y/%m/%d')

    threshold = pd.to_datetime('2014/12/31',format='%Y/%m/%d')
    data_before = meter_data_info.loc[meter_data_info['date_time']<=threshold]
    data_after = meter_data_info.loc[meter_data_info['date_time'] > threshold]

    # before:
    meter_demand_before = data_before.groupby(['id'])['Parking_duration'].count()
    meter_demand_before = meter_demand_before.reset_index()
    num_date = threshold - data_before['date_time'].min()
    days = int(num_date.days) + 1
    meter_demand_before['Parking_duration'] /= days
    meter_demand_before = meter_demand_before.rename(columns = {'Parking_duration':'num_parking_records_per_day_before'})
    meter_demand_before = meter_demand_before.merge(st_data, on = ['id'])
    meter_demand_before = meter_demand_before.rename(columns = {'id':'street_id'})
    meter_demand_before.to_csv('data/street_demand_distribution_before.csv', encoding = 'utf_8_sig', index = False)

    # after:
    meter_demand_after = data_after.groupby(['id'])['Parking_duration'].count()
    meter_demand_after = meter_demand_after.reset_index()
    num_date = data_after['date_time'].max() - threshold
    days = int(num_date.days) - 1
    meter_demand_after['Parking_duration'] /= days
    meter_demand_after = meter_demand_after.rename(columns = {'Parking_duration':'num_parking_records_per_day_after'})
    meter_demand_after = meter_demand_after.merge(st_data, on = ['id'])
    meter_demand_after = meter_demand_after.rename(columns = {'id':'street_id'})
    meter_demand_after.to_csv('data/street_demand_distribution_after.csv', encoding = 'utf_8_sig', index = False)

    #merge to percengtage_change
    meter_demand = meter_demand_before.merge(meter_demand_after[['street_id','num_parking_records_per_day_after']],on = ['street_id'])
    meter_demand['change_per'] = (meter_demand['num_parking_records_per_day_after'] - meter_demand['num_parking_records_per_day_before'])/meter_demand['num_parking_records_per_day_before']
    meter_demand = meter_demand.sort_values(['change_per'],ascending=True)
    meter_demand.to_csv('data/street_demand_distribution_change.csv', encoding='utf_8_sig', index=False)




def generate_area_duration_distribution(meter_data_info,st_data):
    meter_data_info['date_time'] = meter_data_info['PS_year'].astype('str') + '/' + \
                                   meter_data_info['PS_month'].astype('str') + '/' + meter_data_info['PS_day'].astype('str')

    meter_data_info['date_time'] = pd.to_datetime(meter_data_info['date_time'],format='%Y/%m/%d')

    threshold = pd.to_datetime('2014/12/31',format='%Y/%m/%d')
    data_before = meter_data_info.loc[meter_data_info['date_time']<=threshold]
    data_after = meter_data_info.loc[meter_data_info['date_time'] > threshold]

    # before:
    meter_demand_before = data_before.groupby(['id'])['Parking_duration'].mean()
    meter_demand_before = meter_demand_before.reset_index()
    meter_demand_before = meter_demand_before.rename(columns = {'Parking_duration':'Parking_duration_before'})
    meter_demand_before = meter_demand_before.merge(st_data, on = ['id'])
    meter_demand_before = meter_demand_before.rename(columns = {'id':'street_id'})
    meter_demand_before.to_csv('data/street_duration_distribution_before.csv', encoding = 'utf_8_sig', index = False)

    # after:
    meter_demand_after = data_after.groupby(['id'])['Parking_duration'].mean()
    meter_demand_after = meter_demand_after.reset_index()
    meter_demand_after = meter_demand_after.rename(columns = {'Parking_duration':'Parking_duration_after'})
    meter_demand_after = meter_demand_after.merge(st_data, on = ['id'])
    meter_demand_after = meter_demand_after.rename(columns = {'id':'street_id'})
    meter_demand_after.to_csv('data/street_duration_distribution_after.csv', encoding = 'utf_8_sig', index = False)

    #merge to percengtage_change
    meter_demand = meter_demand_before.merge(meter_demand_after[['street_id','Parking_duration_after']],on = ['street_id'])
    meter_demand['change_per'] = (meter_demand['Parking_duration_after'] - meter_demand['Parking_duration_before'])/meter_demand['Parking_duration_before']
    meter_demand = meter_demand.sort_values(['change_per'],ascending=True)
    meter_demand.to_csv('data/street_duration_distribution_change.csv', encoding='utf_8_sig', index=False)


if __name__ == '__main__':
    #############Process st data

    meter_data_info = pd.read_csv('data/meter_data_info.csv')
    st_data = pd.read_csv('data/street_name_area2.csv')
    ############parking volume
    # generate_area_volume_distribution(meter_data_info,st_data)
    ###########3parking duration
    generate_area_duration_distribution(meter_data_info,st_data)