import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from chinese_calendar import is_workday, is_holiday
from rdd import rdd
import datetime
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter
import statsmodels.api as sm

from scipy import stats

colors = ["#3366cc", "#dc3912"]

def star_plot(p_value):
    if p_value >0.1:
        star = ''
    elif  p_value > 0.05 and p_value <= 0.1:
        star = '*'
    elif p_value > 0.01 and p_value <= 0.05:
        star = '**'
    elif p_value <= 0.01:
        star = '***'
    return star


def purpose_bins_parking_vol(data, save_fig=0):
    trip_purpose_list = ['HPS trip', 'LPS trip']
    # trip_purpose_list = ['Working', 'Other']
    subarea_list = [('A-before', 'A-after'), ('B-before', 'A-after')]
    subarea_list_label = ['A-A','B-A']

    colors = sns.color_palette("Paired")
    labels_list = trip_purpose_list.copy()

    for area, area_label in zip(subarea_list,subarea_list_label):
        before_area = area[0]
        after_area = area[1]
        before_mean = []
        after_mean = []
        before_std = []
        after_std = []
        # labels_list = [key[0].replace('-before', '') + '-' + key[1].replace('-after', '') for key in subarea_list]

        data_ks = {}
        p_value_ks = {}
        for purpose in trip_purpose_list:
            data_ks[purpose] = {}
            data_area = data.loc[(data['before_area'] == before_area)&
                                 (data['after_area'] == after_area)]
            if purpose == 'Other':
                data_purpose = data_area.loc[(data_area['Working'] != 1)].copy()
            else:
                data_purpose = data_area.loc[(data_area[purpose] == 1)].copy()
            data_purpose = data_purpose.groupby(['PS_year', 'PS_month', 'PS_day', 'day_of_week'])['Parking_duration'].count()
            data_purpose = data_purpose.reset_index(drop=False)
            data_purpose = data_purpose.rename(columns={'Parking_duration': 'parking_vol'})

            data_ks[purpose]['before'] = np.array(data_purpose.loc[data_purpose['PS_year'] == 2014,'parking_vol'])
            data_ks[purpose]['after'] = np.array(data_purpose.loc[data_purpose['PS_year'] == 2015, 'parking_vol'])
            _, p_value_ks[purpose] = stats.ks_2samp(data_ks[purpose]['before'],data_ks[purpose]['after'] )
            mean_before = data_purpose.loc[data_purpose['PS_year'] == 2014,'parking_vol'].mean()
            std_before = data_purpose.loc[data_purpose['PS_year'] == 2014, 'parking_vol'].std()
            mean_after = data_purpose.loc[data_purpose['PS_year'] == 2015,'parking_vol'].mean()
            std_after = data_purpose.loc[data_purpose['PS_year'] == 2015, 'parking_vol'].std()

            before_mean.append(mean_before)
            after_mean.append(mean_after)
            before_std.append(std_before)
            after_std.append(std_after)



        font_size = 16
        N = len(labels_list)
        ind = np.arange(N)  # the x locations for the groups
        width = 0.3      # the width of the bars
        fig, ax = plt.subplots(figsize=(8, 6))
        rects1 = ax.bar(ind, before_mean, width, color=colors[0])
        rects2 = ax.bar(ind+width, after_mean, width, color=colors[2])

        plt.errorbar(ind, before_mean, yerr=before_std, fmt='k.')
        plt.errorbar(ind+width, after_mean, yerr=after_std, fmt='k.')
        #

        ax.set_ylabel('Parking volume (# veh/day)', fontsize=16)

        y_max = max(max(before_mean) + max(before_std), max(after_mean) + max(after_std))
        y_max = int(np.round(y_max)*1.1)
        y_lim = [0,y_max]


        for loc,y1loc,y2loc,area, in zip(ind,before_mean,after_mean,trip_purpose_list):
            delta = 0.06
            x1 = loc - delta
            x2 = loc + width - delta
            if area_label == 'B-A' and loc == 0:
                y1 = 3 * y1loc
                y2 = 5 * y2loc
                plt.text(x1, y1, str(int(round(y1loc))), fontsize=font_size)
                plt.text(x2, y2, str(int(round(y2loc))), fontsize=font_size)
            else:
                y1 = 0.5 * y1loc
                y2 = 0.5 * y2loc
                plt.text(x1, y1, str(int(round(y1loc))), fontsize=font_size)
                plt.text(x2, y2, str(int(round(y2loc))), fontsize=font_size)
            reduction_per = (y2loc - y1loc) / y1loc
            star = star_plot(p_value_ks[area])
            if reduction_per>0:
                str_temp = '(+'
            else:
                str_temp = '('
            if area_label == 'B-A' and loc == 0:
                plt.text(x2 - delta * 1.1, y2 - 0.05 * y_lim[1], str_temp + str(round(reduction_per * 100, 1)) + '%' + ')',
                         fontsize=font_size)
                plt.text(x2 - delta * 0.2, y2 - 0.1 * y_lim[1], star, fontsize=font_size - 5)
            else:
                plt.text(x2 - delta * 1.1, y2 - 0.05 * y_lim[1], str_temp + str(round(reduction_per * 100, 1)) + '%' + ')',
                         fontsize=font_size)
                plt.text(x2 - delta * 0.2, y2 - 0.1 * y_lim[1], star, fontsize=font_size - 5)
        #
        plt.yticks(fontsize=font_size)
        #

        ax.set_xticks(ind+0.5*width)
        interval = 50
        ax.set_yticks(range(y_lim[0],y_lim[1] + interval, interval))
        ax.set_xticklabels(labels_list,fontsize=16)
        ax.set_xlabel('Inferred trip purposes',fontsize=16)
        labels = ['Before','After']
        ax.legend((rects1[0], rects2[0]) , labels, fontsize=16, loc='upper right')

        x_text = 0.6
        y_text = y_lim[1]*0.9

        plt.ylim(y_lim[0], y_lim[1])

        plt.text(x_text, y_text, area_label, fontsize=font_size * 1.2,
                 bbox=dict(facecolor='red', alpha=0.08))

        plt.tight_layout()
        if save_fig == 0:
            plt.show()
        else:
            name_tail = area_label
            plt.savefig('img/Bins_of_trip_purpose_on_vol_' + name_tail + '.png', dpi=200)




def purpose_bins_parking_duration(data, save_fig=0):
    trip_purpose_list = ['HPS trip', 'LPS trip']
    # trip_purpose_list = ['Working', 'Entertainment']
    colors = sns.color_palette("Paired")
    labels_list = trip_purpose_list.copy()
    subarea_list = [('A-before', 'A-after'), ('B-before', 'A-after')]
    subarea_list_label = ['A-A','B-A']

    for area, area_label in zip(subarea_list,subarea_list_label):
        before_area = area[0]
        after_area = area[1]
        before_mean = []
        after_mean = []
        before_std = []
        after_std = []
        # labels_list = [key[0].replace('-before', '') + '-' + key[1].replace('-after', '') for key in subarea_list]

        data_ks = {}
        p_value_ks = {}

        for purpose in trip_purpose_list:
            data_ks[purpose] = {}
            data_area = data.loc[(data['before_area'] == before_area)&
                                 (data['after_area'] == after_area)]
            if purpose == 'Other':
                data_purpose = data_area.loc[(data_area['Working'] != 1)].copy()
            else:
                data_purpose = data_area.loc[(data_area[purpose] == 1)].copy()
            data_purpose = data_purpose.groupby(['PS_year', 'PS_month', 'PS_day', 'day_of_week'])['Parking_duration'].mean()
            data_purpose = data_purpose.reset_index(drop=False)
            # data_purpose = data_purpose.rename(columns={'Parking_duration': 'parking_vol'})

            data_ks[purpose]['before'] = np.array(data_purpose.loc[data_purpose['PS_year'] == 2014,'Parking_duration'])
            data_ks[purpose]['after'] = np.array(data_purpose.loc[data_purpose['PS_year'] == 2015, 'Parking_duration'])
            _, p_value_ks[purpose] = stats.ks_2samp(data_ks[purpose]['before'],data_ks[purpose]['after'] )
            mean_before = data_purpose.loc[data_purpose['PS_year'] == 2014,'Parking_duration'].mean()
            std_before = data_purpose.loc[data_purpose['PS_year'] == 2014, 'Parking_duration'].std()
            mean_after = data_purpose.loc[data_purpose['PS_year'] == 2015,'Parking_duration'].mean()
            std_after = data_purpose.loc[data_purpose['PS_year'] == 2015, 'Parking_duration'].std()

            before_mean.append(mean_before)
            after_mean.append(mean_after)
            before_std.append(std_before)
            after_std.append(std_after)



        font_size = 16
        N = len(labels_list)
        ind = np.arange(N)  # the x locations for the groups
        width = 0.3      # the width of the bars
        fig, ax = plt.subplots(figsize=(8, 6))
        rects1 = ax.bar(ind, before_mean, width, color=colors[0])
        rects2 = ax.bar(ind+width, after_mean, width, color=colors[2])

        plt.errorbar(ind, before_mean, yerr=before_std, fmt='k.')
        plt.errorbar(ind+width, after_mean, yerr=after_std, fmt='k.')
        #

        ax.set_ylabel('Parking duration (min)', fontsize=16)

        # y_max = max(max(before_mean) + max(before_std), max(after_mean) + max(after_std))
        if area_label == 'B-A':
            y_max = 80
        else:
            y_max = 100
        y_lim = [0,y_max]


        for loc,y1loc,y2loc,area, in zip(ind,before_mean,after_mean,trip_purpose_list):
            delta = 0.06
            x1 = loc - delta
            x2 = loc + width - delta
            y1 = 0.5 * y1loc
            y2 = 0.5 * y2loc
            plt.text(x1, y1, str(int(round(y1loc))), fontsize=font_size)
            plt.text(x2, y2, str(int(round(y2loc))), fontsize=font_size)
            reduction_per = (y2loc - y1loc) / y1loc
            star = star_plot(p_value_ks[area])
            if reduction_per>0:
                str_temp = '(+'
            else:
                str_temp = '('
            plt.text(x2 - delta * 1.1, y2 - 0.05 * y_lim[1], str_temp + str(round(reduction_per * 100, 1)) + '%' + ')',
                     fontsize=font_size)
            plt.text(x2 - delta * 0.2, y2 - 0.1 * y_lim[1], star, fontsize=font_size - 5)
        #
        plt.yticks(fontsize=font_size)
        #

        ax.set_xticks(ind+0.5*width)
        if area_label == 'B-A':
            interval = 15
        else:
            interval = 20
        ax.set_yticks(range(y_lim[0],y_lim[1] + interval, interval))
        ax.set_xticklabels(labels_list,fontsize=16)
        ax.set_xlabel('Areas',fontsize=16)
        labels = ['Before','After']
        ax.legend((rects1[0], rects2[0]) , labels, fontsize=16, loc='upper right')

        x_text = 0.6
        y_text = y_lim[1]*0.9

        plt.text(x_text, y_text, area_label, fontsize=font_size * 1.2,
                 bbox=dict(facecolor='red', alpha=0.08))
        plt.ylim(y_lim[0], y_lim[1])

        plt.tight_layout()
        if save_fig == 0:
            plt.show()
        else:
            name_tail = area_label
            plt.savefig('img/Bins_of_trip_purpose_on_duration_' + name_tail + '.png', dpi=200)




def match_trip_purpose(meter_data_info, St_parking_area_POI):
    St_parking_area_POI_used = St_parking_area_POI.loc[:,['id','Working','Entertainment','Residence','Other','POI_shopping']].copy()

    meter_data_info = meter_data_info.merge(St_parking_area_POI_used, on = ['id'])
    all_shopping_poi = np.array(meter_data_info['POI_shopping'])
    # poi_33_percentile = np.percentile(all_shopping_poi, 33)
    # poi_66_percentile = np.percentile(all_shopping_poi, 66)
    poi_25_percentile = np.percentile(all_shopping_poi, 25)
    poi_75_percentile = np.percentile(all_shopping_poi, 75)
    meter_data_info['HPS trip'] = 0
    meter_data_info['LPS trip'] = 0

    meter_data_info.loc[meter_data_info['POI_shopping']>= poi_75_percentile, 'HPS trip'] = 1
    meter_data_info.loc[meter_data_info['POI_shopping'] <= poi_25_percentile, 'LPS trip'] = 1

    # meter_data_info.loc[meter_data_info['']]
    a=1
    # add time period filter
    #
    # meter_data_info.loc[(meter_data_info['Working'] == 1)&
    #                     ((meter_data_info['PS_hour'] < 7)|(meter_data_info['PS_hour'] > 10)), 'Other'] = 1
    # meter_data_info.loc[(meter_data_info['Working'] == 1)&
    #                     ((meter_data_info['PS_hour'] < 7)|(meter_data_info['PS_hour'] > 10)), 'Working'] = 0
    # process area

    return meter_data_info



if __name__ == '__main__':
    ############data analysis
    meter_data_info = pd.read_csv('data/meter_data_info.csv')
    St_parking_area_POI = pd.read_csv('data/St_parking_area_POI.csv')
    meter_data_info = match_trip_purpose(meter_data_info, St_parking_area_POI)
    purpose_bins_parking_vol(meter_data_info, save_fig=1)
    purpose_bins_parking_duration(meter_data_info, save_fig=1)