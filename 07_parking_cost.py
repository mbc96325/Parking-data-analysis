
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model


from chinese_calendar import is_workday, is_holiday
from rdd import rdd
import datetime

from matplotlib.ticker import FormatStrFormatter
import statsmodels.api as sm
from scipy import stats

labels = ['Before', 'After']
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



def load_survey_data():
    data_2014 = pd.read_csv('data/data_2014_new.csv')
    data_2015 = pd.read_csv('data/data_2015_new.csv')
    # data_201415 = pd.read_csv('data/data_201415.csv')
    # total_2014 = len(data_2014)
    # total_2015 = len(data_2015)
    # print('num 2014', total_2014)
    # print('num 2015', total_2015)
    return data_2014, data_2015


def calculate_cost_meter(data_old, before):
    data = data_old.loc[data_old['Parking_duration']<24*60].copy()

    data['PS_datetime'] = data['PS_year'].apply(str) + '_' + data['PS_month'].apply(str) + '_' +\
                            data['PS_day'].apply(str)+ '_' +data['PS_hour'].apply(str)+ '_' +\
                            data['PS_min'].apply(str)+ '_' +data['PS_sec'].apply(str)

    data['PS_datetime'] = pd.to_datetime(data['PS_datetime'],format = '%Y_%m_%d_%H_%M_%S')

    data['PE_datetime'] = data['PE_year'].apply(str) + '_' + data['PE_month'].apply(str) + '_' +\
                            data['PE_day'].apply(str)+ '_' +data['PE_hour'].apply(str)+ '_' +\
                            data['PE_min'].apply(str)+ '_' +data['PE_sec'].apply(str)
    data['PE_datetime'] = pd.to_datetime(data['PE_datetime'],format = '%Y_%m_%d_%H_%M_%S')

    data['Bill_end_today'] = data['PS_year'].apply(str) + '_' + data['PS_month'].apply(str) + '_' +\
                            data['PS_day'].apply(str)+ '_' +'22_0_0'

    data['Bill_end_today'] = pd.to_datetime(data['Bill_end_today'], format='%Y_%m_%d_%H_%M_%S')
    data['Bill_start_tomorrow'] = pd.to_datetime(data['PS_datetime'].dt.date) + pd.Timedelta('1 days') + pd.Timedelta('7 hours 30 min')
    data['Bill_start_today'] = pd.to_datetime(data['PS_datetime'].dt.date) + pd.Timedelta('7 hours 30 min')

    if before:
        price_unit_A = 3
        price_unit_B = 2.5
        price_C = 5
        unit = 30
        data_daytime = data.loc[(data['PS_datetime'] >= data['Bill_start_today'])&(data['PE_datetime'] <= data['Bill_end_today'])].copy()
        print(len(data),len(data_daytime))
        data_daytime['cost'] = 0
        data_daytime_A = data_daytime.loc[data_daytime['before_area'] == 'A-before'].copy()
        data_daytime_A['cost'] = (data_daytime_A['Parking_duration'] // unit + 1) * price_unit_A
        data_daytime_B = data_daytime.loc[data_daytime['before_area'] == 'B-before'].copy()
        data_daytime_B['cost'] = (data_daytime_B['Parking_duration'] // unit + 1) * price_unit_B
        data_daytime_C = data_daytime.loc[data_daytime['before_area'] == 'C-before'].copy()
        data_daytime_C['cost'] = price_C
        final_data = pd.concat([data_daytime_A,data_daytime_B,data_daytime_C])

    else:
        price_unit_A = 2.5
        price_unit_B = 2
        price_C = 5
        unit = 15
        data_daytime = data.loc[(data['PS_datetime'] >= data['Bill_start_today'])&(data['PE_datetime'] <= data['Bill_end_today'])].copy()
        print(len(data),len(data_daytime))
        data_daytime['cost'] = 0
        data_daytime_A = data_daytime.loc[data_daytime['after_area'] == 'A-after'].copy()
        data_daytime_A['cost'] = (data_daytime_A['Parking_duration'] // unit + 1) * price_unit_A
        data_daytime_B = data_daytime.loc[data_daytime['after_area'] == 'B-after'].copy()
        data_daytime_B['cost'] = (data_daytime_B['Parking_duration'] // unit + 1) * price_unit_B
        data_daytime_C = data_daytime.loc[data_daytime['after_area'] == 'C-after'].copy()
        data_daytime_C['cost'] = price_C
        final_data = pd.concat([data_daytime_A,data_daytime_B,data_daytime_C])
        # data[]
    final_data = final_data.rename(columns={'cost': 'AP_FEE'})
    return final_data

def parking_cost_meter(save_fig, data_2014, data_2015):


    sns.set(font_scale=1.5)
    sns.set_style("white", {"legend.frameon": True})
    fig, ax = plt.subplots(figsize=(10, 5))

    sns.kdeplot(data_2014['AP_FEE'], ax=ax, shade=True, color=colors[0], label=labels[0],bw_adjust=3)
    sns.kdeplot(data_2015['AP_FEE'], ax=ax, shade=True, color=colors[1], label=labels[1],bw_adjust=3)

    meda = data_2014['AP_FEE'].mean()
    plt.axvline(meda, color=colors[0], linestyle='dashed', linewidth=2)
    plt.text(meda-0.5, 0.165, 'Mean = {}'.format(round(meda, 1)),
    horizontalalignment='right', verticalalignment='center',
    fontsize=15, color=colors[0])

    medb = data_2015['AP_FEE'].mean()
    plt.axvline(medb, color=colors[1], linestyle='dashed', linewidth=2)
    plt.text(medb+0.5, 0.165, 'Mean = {}'.format(round(medb, 1)),
    horizontalalignment='left', verticalalignment='center',
    fontsize=15, color=colors[1])
    plt.xlim(0, 30)
    new_ticks2 = list(np.arange(0,0.18,0.03))
    plt.ylim(0, 0.18)
    #new_ticks.append(30)
    plt.yticks(new_ticks2)
    #new_ticks = list(range(0,400,60))
    #new_ticks.append(30)
    #plt.xticks(new_ticks)
    plt.xlabel('Parking cost per trip (RMB)', fontsize=16)
    plt.ylabel('Density', fontsize=16)
    plt.tight_layout()

    if save_fig == 0:
        plt.show()
    else:
        plt.savefig('img/actual_price_2014_2015_meter.png', dpi=300)



def parking_duration_meter(save_fig, single_day_data, data_2014, data_2015):
    if single_day_data:
        labels_new = ['Before (single day)','After (single day)']
    else:
        labels_new = ['Before (4 months)', 'After (8 months)']

    sns.set(font_scale=1.5)
    sns.set_style("white", {"legend.frameon": True})
    fig, ax = plt.subplots(figsize=(10, 5))
    if single_day_data:
        data_2014 = data_2014.loc[(data_2014['PS_year'] == 2014)&(data_2014['PS_month'] == 9)&(data_2014['PS_day'] == 12)]
        data_2015 = data_2015.loc[(data_2015['PS_year'] == 2015)&(data_2015['PS_month'] == 8)&(data_2015['PS_day'] == 7)]
    if single_day_data:
        sns.kdeplot(data_2014['Parking_duration'], ax=ax, shade=True, color=colors[0], label=labels_new[0],bw_adjust=0.3)
        sns.kdeplot(data_2015['Parking_duration'], ax=ax, shade=True, color=colors[1], label=labels_new[1],bw_adjust=0.3)
    else:
        sns.kdeplot(data_2014['Parking_duration'], ax=ax, shade=True, color=colors[0], label=labels_new[0],bw_adjust=0.4)#0.4
        sns.kdeplot(data_2015['Parking_duration'], ax=ax, shade=True, color=colors[1], label=labels_new[1],bw_adjust=0.4)#0.4
    # sns.histplot(data_2014['Parking_duration'], ax=ax, binwidth = 5, color=colors[0])
    # sns.histplot(data_2015['Parking_duration'], ax=ax, binwidth = 5, color=colors[1])


    #
    x_lim = 240
    unit = 30
    # if single_day_data:
    intervals = range(unit,x_lim,unit)
    for int_ in intervals:
        plt.axvline(int_, color='k', linestyle='dashed', linewidth=1)

    intervals = range(15,x_lim,unit)
    for int_ in intervals:
        plt.axvline(int_, color='gray', linestyle='dashed', linewidth=0.8)

    if not single_day_data:
        meda = data_2014['Parking_duration'].mean()
        plt.axvline(meda, color=colors[0], linestyle='dashed', linewidth=2)
        plt.text(meda+1, 0.020, 'Mean = {}'.format(round(meda, 1)),
                 horizontalalignment='left', verticalalignment='center',
                 fontsize=15, color=colors[0])

        medb = data_2015['Parking_duration'].mean()

        plt.axvline(medb, color=colors[1], linestyle='dashed', linewidth=2)
        plt.text(medb-1, 0.020, 'Mean = {}'.format(round(medb, 1)),
        horizontalalignment='right', verticalalignment='center',
        fontsize=15, color=colors[1])

    plt.xlim(0, x_lim)
    # new_ticks2 = list(np.arange(0,0.18,0.03))
    # plt.ylim(0, 0.18)
    # #new_ticks.append(30)
    # plt.yticks(new_ticks2)

    new_ticks = list(range(0,x_lim + unit,30))
    plt.xticks(new_ticks)
    plt.xlabel('Parking duration per trip (min)', fontsize=16)
    plt.ylabel('Density', fontsize=16)
    plt.legend()
    plt.tight_layout()
    if save_fig == 0:
        plt.show()
    else:
        if single_day_data:
            plt.savefig('img/actual_parking_duration_2014_2015_meter_single_day.jpg', dpi=300)
        else:
            plt.savefig('img/actual_parking_duration_2014_2015_meter_full_data.jpg', dpi=300)




def parking_cost_survey(save_fig):
    data_2014, data_2015 = load_survey_data()


    #---------------------------------------------
    sns.set(font_scale=1.5)
    sns.set_style("white", {"legend.frameon": True})
    fig, ax = plt.subplots(figsize=(10, 5))

    sns.kdeplot(data_2014['AP_FEE'], ax=ax, shade=True, color=colors[0], label=labels[0])
    sns.kdeplot(data_2015['AP_FEE'], ax=ax, shade=True, color=colors[1], label=labels[1])

    meda = data_2014['AP_FEE'].mean()
    plt.axvline(meda, color=colors[0], linestyle='dashed', linewidth=2)
    plt.text(meda-0.5, 0.165, 'Mean = {}'.format(round(meda, 1)),
    horizontalalignment='right', verticalalignment='center',
    fontsize=15, color=colors[0])

    medb = data_2015['AP_FEE'].mean()
    plt.axvline(medb, color=colors[1], linestyle='dashed', linewidth=2)
    plt.text(medb+0.5, 0.165, 'Mean = {}'.format(round(medb, 1)),
    horizontalalignment='left', verticalalignment='center',
    fontsize=15, color=colors[1])
    plt.xlim(0, 30)
    new_ticks2 = list(np.arange(0,0.18,0.03))
    plt.ylim(0, 0.18)
    #new_ticks.append(30)
    plt.yticks(new_ticks2)
    #new_ticks = list(range(0,400,60))
    #new_ticks.append(30)
    #plt.xticks(new_ticks)
    plt.xlabel('Actual Total Parking Fee (RMB)', fontsize=16)
    plt.ylabel('Density', fontsize=16)
    plt.tight_layout()

    if save_fig == 0:
        plt.show()
    else:
        plt.savefig('img/actual_price_2014_2015_survey.png', dpi=300)



def bins_subarea(data, day_of_week_list, subarea_list, save_fig=0):
    for day_list in day_of_week_list:
    #colors = ["black", "gray"]
        colors = sns.color_palette("Paired")
        before_mean = []
        after_mean = []
        before_std =[]
        after_std = []
        labels_list = [key[0].replace('-before', '') + '-' + key[1].replace('-after', '') for key in subarea_list]
        data_ks = {}
        p_value_ks = {}
        for area in subarea_list:
            data_ks[area] = {}
            before_area = area[0]
            after_area = area[1]
            data_area = data.loc[(data['before_area'] == before_area) &
                                  (data['after_area'] == after_area)]

            data_area = data_area.groupby(['PS_year', 'PS_month', 'PS_day', 'day_of_week'])['AP_FEE'].sum()
            data_area = data_area.reset_index(drop=False)
            data_area = data_area.loc[data_area['day_of_week'].isin(day_list)]
            data_area = data_area.rename(columns={'AP_FEE': 'parking_vol'})

            data_ks[area]['before'] = np.array(data_area.loc[data_area['PS_year'] == 2014,'parking_vol'])
            data_ks[area]['after'] = np.array(data_area.loc[data_area['PS_year'] == 2015, 'parking_vol'])
            _, p_value_ks[area] = stats.ks_2samp(data_ks[area]['before'],data_ks[area]['after'] )
            mean_before = data_area.loc[data_area['PS_year'] == 2014,'parking_vol'].mean()
            std_before = data_area.loc[data_area['PS_year'] == 2014, 'parking_vol'].std()
            mean_after = data_area.loc[data_area['PS_year'] == 2015,'parking_vol'].mean()
            std_after = data_area.loc[data_area['PS_year'] == 2015, 'parking_vol'].std()

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

        ax.set_ylabel('Total daily parking cost (RMB/day)', fontsize=16)
        y_lim = [0,9000]
        plt.ylim(y_lim[0], y_lim[1])

        for loc,y1loc,y2loc,area, in zip(ind,before_mean,after_mean,subarea_list):
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
        interval = 2000
        ax.set_yticks(range(y_lim[0],y_lim[1], interval))
        ax.set_xticklabels(labels_list,fontsize=16)
        ax.set_xlabel('Areas',fontsize=16)
        labels = ['Before','After']
        ax.legend((rects1[0], rects2[0]) , labels, fontsize=16, loc='upper right')
        plt.tight_layout()
        if save_fig == 0:
            plt.show()
        else:
            day_of_week_str = [str(i) for i in day_list]
            name_tail = '_'.join(day_of_week_str)
            plt.savefig('img/Bins_of_parking_cost_' + name_tail + '.png', dpi=200)


def plot_total_daily_cost(save_fig, data_2014, data_2015):
    # ################################ time of day distribution
    subarea_list = [('A-before', 'A-after'), ('B-before', 'A-after')]
    day_of_week_list = [[0, 1, 2, 3, 4], [5, 6]]
    meter_data_info = pd.concat([data_2014, data_2015])
    bins_subarea(meter_data_info, day_of_week_list, subarea_list, save_fig=save_fig)

if __name__ == '__main__':
    save_fig = 1


    meter_data_info = pd.read_csv('data/meter_data_info.csv')
    data_2014 = meter_data_info.loc[meter_data_info['PS_year'] == 2014]
    data_2015 = meter_data_info.loc[meter_data_info['PS_year'] == 2015]

    data_2014 = calculate_cost_meter(data_2014, before = True)
    data_2015 = calculate_cost_meter(data_2015, before = False)
    ##############################
    # parking_cost_meter(save_fig, data_2014, data_2015)

    single_day_data_list = [False] #
    for single_day_data in single_day_data_list:
        parking_duration_meter(save_fig, single_day_data, data_2014, data_2015)
    ##############################
    # plot_total_daily_cost(save_fig, data_2014, data_2015)
