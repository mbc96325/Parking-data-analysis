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


def from_dayid_to_date(x_ticks,data_vol_city):
    new_ticks = []
    for day_id in x_ticks:
        year = data_vol_city.loc[data_vol_city['day_id']==day_id,'PS_year'].values[0]
        month = data_vol_city.loc[data_vol_city['day_id'] == day_id, 'PS_month'].values[0]
        date = data_vol_city.loc[data_vol_city['day_id'] == day_id, 'PS_day'].values[0]
        str_date = str(year).replace('20','') + '/'+ str(month) +'/'+ str(date)
        new_ticks.append(str_date)
    return new_ticks


def regression_two_slop(data_rdd,threshold, fix_before_zero):
    data_rdd['D'] = 0
    data_rdd.loc[data_rdd['day_id'] >= threshold, 'D'] = 1
    data_rdd['intercept'] = 1
    data_rdd['D_times_day_id'] = data_rdd['D'] * data_rdd['day_id']
    x_before = data_rdd.loc[data_rdd['day_id']<threshold,'day_id']
    x_after = data_rdd.loc[data_rdd['day_id'] >= threshold,'day_id']
    # data_rdd_before = data_rdd.loc[data_rdd['day_id']<threshold]
    # data_rdd_after = data_rdd.loc[data_rdd['day_id']  >= threshold]
    if fix_before_zero:
        x_col = ['intercept', 'D', 'D_times_day_id']
    else:
        x_col = ['intercept','day_id','D','D_times_day_id']

    y_col = ['parking_vol']
    mod = sm.OLS( data_rdd.loc[:,y_col], data_rdd.loc[:,x_col])

    res = mod.fit()
    para = res.params
    if fix_before_zero:
        y_before = para.intercept * 1 + 0 * np.array(x_before) + para.D * 0 + para.D_times_day_id * 0
        y_after = para.intercept * 1 + 0 * np.array(x_after) + para.D * 1 + para.D_times_day_id * np.array(x_after)
    else:
        y_before = para.intercept * 1 + para.day_id * np.array(x_before) + para.D * 0 + para.D_times_day_id * 0
        y_after = para.intercept * 1 + para.day_id * np.array(x_after) + para.D * 1 + para.D_times_day_id * np.array(x_after)

    p_treated_1 = res.pvalues.D
    p_treated_2 = res.pvalues.D_times_day_id
    if fix_before_zero:
        TREATED_effect = (para.intercept * 1 + 0 * threshold + para.D * 0 + para.D_times_day_id * 0) - \
                         (para.intercept * 1 + 0 * (threshold+1) + para.D * 1 + para.D_times_day_id * (threshold+1))
    else:
        TREATED_effect = (para.intercept * 1 + para.day_id * threshold + para.D * 0 + para.D_times_day_id * 0) - \
                         (para.intercept * 1 + para.day_id * (threshold+1) + para.D * 1 + para.D_times_day_id * (threshold+1))

    print(res.summary())

    a=1
    return x_before, x_after,y_before,y_after,TREATED_effect, p_treated_1, p_treated_2, para.D_times_day_id, para.intercept


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



def RD_design_city_level(data, duration_or_volume, save_fig, fix_before_zero):
    if duration_or_volume == 'volume':
        data_vol = data.groupby(['PS_year','PS_month','PS_day','day_of_week','before_area','after_area'])['id'].count()
        data_vol = data_vol.reset_index(drop=False)
        data_vol = data_vol.rename(columns={'id': 'parking_vol'})
        data_vol_city = data_vol.groupby(['PS_year', 'PS_month', 'PS_day', 'day_of_week'])['parking_vol'].sum()
        data_vol_city = data_vol_city.reset_index(drop=False)
    else:
        data_vol_city = data.groupby(['PS_year','PS_month','PS_day','day_of_week'])['Parking_duration'].mean()
        data_vol_city = data_vol_city.reset_index(drop=False)
        data_vol_city = data_vol_city.rename(columns={'Parking_duration': 'parking_vol'})




    data_vol_city['day_id'] = range(0,len(data_vol_city))
    threshold = data_vol_city.loc[(data_vol_city['PS_year']==2015)&(data_vol_city['PS_month']==1)&(data_vol_city['PS_day']==1), 'day_id'].values[0]

    x_ticks = data_vol_city.loc[(data_vol_city['PS_day']==1)&(data_vol_city['PS_month'].isin([1,3,5,7,9,11])),'day_id']
    x_ticks = x_ticks.append(data_vol_city.loc[(data_vol_city['PS_day']==31)&(data_vol_city['PS_month']==8)&(data_vol_city['PS_year']==2015),'day_id'])

    new_ticks = from_dayid_to_date(x_ticks,data_vol_city)

    data_vol_city = delete_holidays(data_vol_city)


    x_name = 'day_id'
    y_name = 'parking_vol'
    # truncate
    # bandwidth_opt = rdd.optimal_bandwidth(data_vol_city['parking_vol'], data_vol_city['day_id'], cut=threshold)
    # print("Optimal bandwidth:", bandwidth_opt)
    # data_rdd = rdd.truncated_data(data_vol_city, 'day_id', bandwidth_opt, cut=threshold)

    # truncate
    # bandwidth = 3*30
    # data_rdd = rdd.truncated_data(data_vol_city, 'day_id', bandwidth, cut=threshold)


    # No truncate
    data_rdd = data_vol_city.copy()

    # check
    # plt.figure(figsize=(12, 8))
    # plt.scatter(data_rdd['day_id'], data_rdd['parking_vol'], facecolors='none', edgecolors='r')
    # plt.xlabel('Day')
    # plt.ylabel('Parking volume')
    # plt.axvline(x=threshold, color='b')
    # plt.show()
    # plt.close()

    # too much noise need to bin the data
    num_bin =120
    data_binned = rdd.bin_data(data_rdd, 'parking_vol', 'day_id', num_bin)

    # estimation========================RDD
    # model = rdd.rdd(data_rdd, 'day_id', 'parking_vol', cut=threshold)
    # results = model.fit()
    # print(results.summary())
    #
    # # get fitting line
    # para = results.params
    #
    # x_before = data_rdd.loc[data_rdd['day_id']<threshold,'day_id']
    # x_after = data_rdd.loc[data_rdd['day_id'] >= threshold,'day_id']
    #
    # y_before = para.Intercept * 1 + para.day_id * np.array(x_before) + para.TREATED * 0
    # y_after = para.Intercept * 1 + para.day_id * np.array(x_after) + para.TREATED * 1
    # p_treated = results.pvalues.TREATED
    x_lim = [x_ticks.iloc[0], x_ticks.iloc[-1]]
    if duration_or_volume == 'volume':
        y_lim = [0,2200]
    else:
        y_lim = [0,100]
    name_tail = 'city_level'
    regression_and_plot(data_rdd, threshold, fix_before_zero, x_ticks, new_ticks, x_lim, y_lim, save_fig, name_tail)


def regression_and_plot(data_rdd, threshold, fix_before_zero, x_ticks, new_ticks, x_lim, y_lim, save_fig, name_tail):
    # estimation========================linear reg

    x_before, x_after, y_before, y_after, TREATED_effect, p_treated_1, p_treated_2, beta_3, intercept = regression_two_slop(data_rdd,threshold, fix_before_zero)
    TREATED_effect = -TREATED_effect
    # plot
    font_size = 18
    plt.figure(figsize=(10, 8))
    # plt.scatter(data_binned['day_id'], data_binned['parking_vol'],
    #     s = data_binned['n_obs'], facecolors='none', edgecolors='r')
    plt.scatter(data_rdd['day_id'], data_rdd['parking_vol'], marker ='.',s=15)
    plt.axvline(x=threshold, color='b')
    plt.plot(x_before, y_before, 'k')
    plt.plot(x_after, y_after, 'k')
    plt.xlabel('Time',fontsize=font_size)
    if duration_or_volume == 'volume':
        plt.ylabel('Parking volume (# veh/day)',fontsize=font_size)
    else:
        plt.ylabel('Parking duration (min)', fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.xticks(x_ticks,new_ticks,fontsize=font_size)

    plt.xlim(x_lim[0],x_lim[1])

    plt.ylim(y_lim[0],y_lim[1])
    x_text =  (x_lim[1] - x_lim[0])* 0.6 + x_lim[0]
    y_text = (y_lim[1] - y_lim[0]) * 0.09 + y_lim[0]
    y_text2 = (y_lim[1] - y_lim[0]) * 0.04 + y_lim[0]

    star_1 = star_plot(p_treated_1)
    star_2 = star_plot(p_treated_2)
    if duration_or_volume == 'volume':
        plt.text(x_text, y_text, 'ITE: ' + str(round(TREATED_effect,1)) + '$^{' + star_1 +'}$'+ ' (' + str(round(TREATED_effect/intercept*100,1))+ '%)' , fontsize=font_size*1.1)
        plt.text(x_text, y_text2, 'TES: ' + str(round(beta_3, 2)) + '$^{' + star_2 + '}$',
             fontsize=font_size * 1.1)

    else:
        plt.text(x_text, y_text, 'ITE: ' + str(round(TREATED_effect,1)) + '$^{' + star_1 +'}$'+ ' (' + str(round(TREATED_effect/intercept*100,1))+ '%)' , fontsize=font_size*1.1)
        plt.text(x_text, y_text2, 'TES: ' + str(round(beta_3, 2)) + '$^{' + star_2 + '}$',
             fontsize=font_size * 1.1)
    # plt.text(x_text, y_text2, "p-value: {:.3f}".format(p_treated), fontsize=font_size * 1.1)
    plt.tight_layout()

    if save_fig==1:
        plt.savefig('img/RDD_' + name_tail + '_' + duration_or_volume + '.png', dpi=200)
    else:
        plt.show()



def RD_design_city_level_weekdays_weekends(data,duration_or_volume, day_of_week_list, save_fig, fix_before_zero):


    for key in day_of_week_list:
        if duration_or_volume == 'volume':
            day_of_week = key
            data_vol = data.groupby(['PS_year', 'PS_month', 'PS_day', 'day_of_week', 'before_area', 'after_area'])[
                'id'].count()
            data_vol = data_vol.reset_index(drop=False)
            data_vol = data_vol.rename(columns={'id': 'parking_vol'})
            # filter

            ###Cal threshold
            data_vol_city = data_vol.groupby(['PS_year', 'PS_month', 'PS_day', 'day_of_week'])['parking_vol'].sum()
            data_vol_city = data_vol_city.reset_index(drop=False)
        else:
            day_of_week = key
            data_vol_city = data.groupby(['PS_year', 'PS_month', 'PS_day', 'day_of_week'])[
                'Parking_duration'].mean()
            data_vol_city = data_vol_city.reset_index(drop=False)
            data_vol_city = data_vol_city.rename(columns={'Parking_duration': 'parking_vol'})
            # filter




        data_vol_city['day_id'] = range(0,len(data_vol_city))
        threshold = data_vol_city.loc[(data_vol_city['PS_year']==2015)&(data_vol_city['PS_month']==1)&(data_vol_city['PS_day']==1), 'day_id'].values[0]

        ##Cal threshold
        x_ticks = data_vol_city.loc[
            (data_vol_city['PS_day'] == 1) & (data_vol_city['PS_month'].isin([1, 3, 5, 7, 9, 11])), 'day_id']
        x_ticks = x_ticks.append(data_vol_city.loc[
                                     (data_vol_city['PS_day'] == 31) & (data_vol_city['PS_month'] == 8) & (
                                                 data_vol_city['PS_year'] == 2015), 'day_id'])

        new_ticks = from_dayid_to_date(x_ticks,data_vol_city)

        data_vol_city = delete_holidays(data_vol_city)

        data_vol_city = data_vol_city.loc[data_vol_city['day_of_week'].isin(day_of_week)]


        x_name = 'day_id'
        y_name = 'parking_vol'
        # truncate
        # bandwidth_opt = rdd.optimal_bandwidth(data_vol_city['parking_vol'], data_vol_city['day_id'], cut=threshold)
        # print("Optimal bandwidth:", bandwidth_opt)
        # data_rdd = rdd.truncated_data(data_vol_city, 'day_id', bandwidth_opt, cut=threshold)

        # truncate
        # bandwidth = 3*30
        # data_rdd = rdd.truncated_data(data_vol_city, 'day_id', bandwidth, cut=threshold)


        # No truncate
        data_rdd = data_vol_city.copy()

        # check
        # plt.figure(figsize=(12, 8))
        # plt.scatter(data_rdd['day_id'], data_rdd['parking_vol'], facecolors='none', edgecolors='r')
        # plt.xlabel('Day')
        # plt.ylabel('Parking volume')
        # plt.axvline(x=threshold, color='b')
        # plt.show()
        # plt.close()

        # too much noise need to bin the data
        num_bin =120
        data_binned = rdd.bin_data(data_rdd, 'parking_vol', 'day_id', num_bin)

        # estimation
        # model = rdd.rdd(data_rdd, 'day_id', 'parking_vol', cut=threshold)
        # results = model.fit()
        # print(results.summary())
        #
        # # get fitting line
        # para = results.params
        x_lim = [x_ticks.iloc[0], x_ticks.iloc[-1]]
        # x_before = np.arange(0, threshold)
        # x_after = np.arange(threshold, x_lim[1],1)
        #
        #
        # y_before = para.Intercept * 1 + para.day_id * np.array(x_before) + para.TREATED * 0
        # y_after = para.Intercept * 1 + para.day_id * np.array(x_after) + para.TREATED * 1

        # estimation========================linear reg
        if duration_or_volume == 'volume':
            y_lim = [0,2200]
        else:
            y_lim = [0, 100]

        if 1 in day_of_week:
            name_tail = 'weekday'
        else:
            name_tail = 'weekend'
        regression_and_plot(data_rdd, threshold, fix_before_zero, x_ticks, new_ticks, x_lim, y_lim, save_fig, name_tail)


def RD_design_subarea_level(data, duration_or_volume, subarea_list, save_fig):
    for key in subarea_list:
        before_area = key[0]
        after_area = key[1]

        if duration_or_volume == 'volume':

            data_vol = data.groupby(['PS_year','PS_month','PS_day','day_of_week','before_area','after_area'])['id'].count()
            data_vol = data_vol.reset_index(drop=False)
            data_vol = data_vol.rename(columns = {'id':'parking_vol'})
            # filter
            data_vol = data_vol.loc[(data_vol['before_area'] == before_area)&(data_vol['after_area'] == after_area)]
            #
            data_vol_city = data_vol.groupby(['PS_year','PS_month','PS_day','day_of_week'])['parking_vol'].sum()
            data_vol_city = data_vol_city.reset_index(drop=False)


        else:
            data_vol = data.loc[(data['before_area'] == before_area) & (data['after_area'] == after_area)]
            data_vol_city = data_vol.groupby(['PS_year','PS_month','PS_day','day_of_week'])['Parking_duration'].mean()
            data_vol_city = data_vol_city.reset_index(drop=False)
            data_vol_city = data_vol_city.rename(columns={'Parking_duration': 'parking_vol'})

        data_vol_city['day_id'] = range(0,len(data_vol_city))


        threshold = data_vol_city.loc[(data_vol_city['PS_year']==2015)&(data_vol_city['PS_month']==1)&(data_vol_city['PS_day']==1), 'day_id'].values[0]

        x_ticks = data_vol_city.loc[
            (data_vol_city['PS_day'] == 1) & (data_vol_city['PS_month'].isin([1, 3, 5, 7, 9, 11])), 'day_id']
        x_ticks = x_ticks.append(data_vol_city.loc[
                                     (data_vol_city['PS_day'] == 31) & (data_vol_city['PS_month'] == 8) & (
                                                 data_vol_city['PS_year'] == 2015), 'day_id'])

        new_ticks = from_dayid_to_date(x_ticks,data_vol_city)

        data_vol_city = delete_holidays(data_vol_city)


        # truncate
        # bandwidth_opt = rdd.optimal_bandwidth(data_vol_city['parking_vol'], data_vol_city['day_id'], cut=threshold)
        # print("Optimal bandwidth:", bandwidth_opt)
        # data_rdd = rdd.truncated_data(data_vol_city, 'day_id', bandwidth_opt, cut=threshold)

        # truncate
        # bandwidth = 3*30
        # data_rdd = rdd.truncated_data(data_vol_city, 'day_id', bandwidth, cut=threshold)


        # No truncate
        data_rdd = data_vol_city.copy()

        # check
        # plt.figure(figsize=(12, 8))
        # plt.scatter(data_rdd['day_id'], data_rdd['parking_vol'], facecolors='none', edgecolors='r')
        # plt.xlabel('Day')
        # plt.ylabel('Parking volume')
        # plt.axvline(x=threshold, color='b')
        # plt.show()
        # plt.close()

        # too much noise need to bin the data
        num_bin =120
        data_binned = rdd.bin_data(data_rdd, 'parking_vol', 'day_id', num_bin)

        # ========================estimation RDD
        # model = rdd.rdd(data_rdd, 'day_id', 'parking_vol', cut=threshold)
        # results = model.fit()
        # print(results.summary())
        #
        # # get fitting line
        # para = results.params
        #
        # x_before = np.arange(0, threshold)
        # x_after = np.arange(threshold, x_lim[1],1)
        #
        # y_before = para.Intercept * 1 + para.day_id * np.array(x_before) + para.TREATED * 0
        # y_after = para.Intercept * 1 + para.day_id * np.array(x_after) + para.TREATED * 1


        # estimation========================linear reg RDD
        x_lim = [data_rdd['day_id'].min(), data_rdd['day_id'].max()]
        if duration_or_volume == 'volume':
            if before_area == 'C-before':
                y_lim = [0,250]
            else:
                y_lim = [0, data_rdd['parking_vol'].max() * 1.2]
        else:
            if before_area != 'C-before':
                y_lim = [0, 100]
            else:
                y_lim = [0, data_rdd['parking_vol'].max() * 1.2]

        name_tail = before_area + '_' + after_area
        regression_and_plot(data_rdd, threshold, fix_before_zero, x_ticks, new_ticks, x_lim, y_lim, save_fig, name_tail)


def time_of_day_distribution(data, day_of_week_list, month_dict, duration_or_volume, save_fig):
    if duration_or_volume == 'duration':
        data = data.loc[data['Parking_duration']<= 8*60] # drop wrong data

    for day_of_week in day_of_week_list:
        if duration_or_volume == 'duration':
            fig, ax1 = plt.subplots(figsize=(8, 6))
        else:
            fig, ax1 = plt.subplots(figsize=(8, 6))
        font_size = 17
        if duration_or_volume == 'duration':
            ax1.set_ylabel('Parking duration (min)', fontsize=font_size)
        else:
            ax1.set_ylabel('Parking volume (# veh/hour)', fontsize=font_size)
        ax1.set_xlabel('Time of day', fontsize=font_size)
        color_id = 1
        colors = sns.color_palette("Paired")
        if  duration_or_volume == 'duration':
            hour_lim = [8, 21]
        else:
            hour_lim = [8, 22]
        for name in month_dict:
            year_month = month_dict[name]
            data_month = data.loc[(data['PS_year'] == year_month[0])&
                                  (data['PS_month'].isin(year_month[1]))&
                                  (data['day_of_week'].isin(day_of_week))]


            data_month = data_month.loc[(data_month['PS_hour']>=hour_lim[0])&
                                        (data_month['PS_hour']<=hour_lim[1])]

            if duration_or_volume == 'duration':
                # data_month['delete'] = 0
                # data_month.loc[(data_month['PS_hour'] == 22) & (data_month['PS_min']>=15), 'delete'] = 1
                # data_month = data_month.loc[data_month['delete'] == 0]
                data_month_vol = data_month.groupby(['PS_year', 'PS_month', 'PS_day', 'day_of_week', 'PS_hour'])[
                    'Parking_duration'].mean()
                data_month_vol = data_month_vol.reset_index(drop=False)
                data_month_vol = data_month_vol.groupby(['PS_hour'])['Parking_duration'].agg(['mean','std'])
                data_month_vol_mean_std = data_month_vol.reset_index(drop=False)

            else:
                data_month_vol = data_month.groupby(['PS_year','PS_month','PS_day','day_of_week','PS_hour'])['id'].count()
                data_month_vol = data_month_vol.reset_index(drop=False)
                data_month_vol = data_month_vol.rename(columns={'id': 'parking_vol'})

                # get mean and std
                data_month_vol_mean_std = data_month_vol.groupby(['PS_hour'])['parking_vol'].agg(['mean','std'])
                data_month_vol_mean_std = data_month_vol_mean_std.reset_index(drop=False)

            x = data_month_vol_mean_std['PS_hour']
            y = data_month_vol_mean_std['mean']
            error_bar = data_month_vol_mean_std['std']


            ax1.plot(x, y, color=colors[color_id], linewidth=1.5, label=name, marker = '^', markersize = 8) #

            meanst = y
            if duration_or_volume == 'duration':
                sdt = error_bar * 0.5
            else:
                sdt = error_bar * 0.5

            # if duration_or_volume != 'duration':
            ax1.fill_between(x, meanst-sdt, meanst+sdt, alpha=0.1, edgecolor=colors[color_id], facecolor=colors[color_id])

            color_id += 2

        ax1.tick_params(axis='y', labelsize=font_size)
        ax1.tick_params(axis='x', labelsize=font_size)
        ax1.legend(fontsize=font_size*0.9)
        x_ticks = range(hour_lim[0],hour_lim[1] + 2,2)
        if duration_or_volume == 'duration':
            new_ticks = [str(i) for i in x_ticks]
        else:
            new_ticks = [str(i) + ':00' for i in x_ticks]
        plt.xticks(x_ticks,new_ticks,fontsize=font_size)
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        if duration_or_volume == 'duration':
            plt.ylim([20, 90])
        else:
            plt.ylim([0,180])
        if save_fig == 0:
            plt.show()
        else:
            day_of_week_str = [str(i) for i in day_of_week]
            name_tail = '_'.join(day_of_week_str)
            if duration_or_volume == 'duration':
                plt.savefig('img/Time_of_day_duration' + name_tail + '.png', dpi=200)
            else:
                plt.savefig('img/Time_of_day_' + name_tail + '.png', dpi=200)


def time_of_day_distribution_subarea(data,subarea_list, day_of_week_list, month_dict,duration_or_volume, save_fig):


    if duration_or_volume == 'duration':
        data = data.loc[data['Parking_duration']<= 8*60] # drop wrong data
    for area in subarea_list:
        before_area = area[0]
        after_area = area[1]
        for day_of_week in day_of_week_list:
            if duration_or_volume =='duration':
                fig, ax1 = plt.subplots(figsize=(8, 6))
            else:
                fig, ax1 = plt.subplots(figsize=(8, 6))
            font_size = 17
            if duration_or_volume == 'duration':
                ax1.set_ylabel('Parking duration (min)', fontsize=font_size)
            else:
                ax1.set_ylabel('Parking volume (Veh/hour)', fontsize=font_size)
            ax1.set_xlabel('Time of day', fontsize=font_size)
            color_id = 1
            colors = sns.color_palette("Paired")
            if duration_or_volume == 'duration':
                hour_lim = [8, 21]
            else:
                hour_lim = [8, 22]
            for name in month_dict:
                year_month = month_dict[name]
                data_month = data.loc[(data['PS_year'] == year_month[0])&
                                      (data['PS_month'].isin(year_month[1]))&
                                      (data['day_of_week'].isin(day_of_week))&
                                      (data['before_area'] == before_area)&
                                      (data['after_area'] == after_area)]


                data_month = data_month.loc[(data_month['PS_hour']>=hour_lim[0])&
                                            (data_month['PS_hour']<=hour_lim[1])]
                if duration_or_volume == 'duration':
                    data_month_vol = data_month.groupby(['PS_year','PS_month','PS_day','day_of_week','PS_hour'])['Parking_duration'].mean()
                    data_month_vol = data_month_vol.reset_index(drop=False)
                    data_month_vol = data_month_vol.groupby(['PS_hour'])['Parking_duration'].agg(['mean', 'std'])
                    data_month_vol_mean_std = data_month_vol.reset_index(drop=False)
                else:
                    data_month_vol = data_month.groupby(['PS_year','PS_month','PS_day','day_of_week','PS_hour'])['id'].count()
                    data_month_vol = data_month_vol.reset_index(drop=False)
                    data_month_vol = data_month_vol.rename(columns={'id': 'parking_vol'})

                    # get mean and std
                    data_month_vol_mean_std = data_month_vol.groupby(['PS_hour'])['parking_vol'].agg(['mean','std'])
                    data_month_vol_mean_std = data_month_vol_mean_std.reset_index(drop=False)

                x = data_month_vol_mean_std['PS_hour']
                y = data_month_vol_mean_std['mean']
                error_bar = data_month_vol_mean_std['std']

                ax1.plot(x, y, color=colors[color_id], linewidth=1.5, label=name, marker = '^', markersize = 8) #

                meanst = y
                sdt = error_bar*0.5
                # if duration_or_volume != 'duration':
                ax1.fill_between(x, meanst-sdt, meanst+sdt, alpha=0.1, edgecolor=colors[color_id], facecolor=colors[color_id])
                ax1.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
                color_id += 2

            ax1.tick_params(axis='y', labelsize=font_size)
            ax1.tick_params(axis='x', labelsize=font_size)
            ax1.legend(fontsize=font_size*0.9)
            x_ticks = range(hour_lim[0],hour_lim[1]+1 + 2,2)
            if duration_or_volume == 'duration':
                new_ticks = [str(i) + '' for i in x_ticks]
            else:
                new_ticks = [str(i) + ':00' for i in x_ticks]
            plt.xticks(x_ticks,new_ticks,fontsize=font_size)
            fig.tight_layout()  # otherwise the right y-label is slightly clipped
            if duration_or_volume == 'duration':
                y_lim = [30, 80]
                if before_area == 'A-before' and after_area == 'A-after':
                    name_area = 'A-A'
                elif before_area == 'B-before' and after_area == 'A-after':
                    name_area = 'B-A'
                elif before_area == 'C-before' and after_area == 'A-after':
                    name_area = 'C-A'
            else:
                if before_area == 'A-before' and after_area == 'A-after':
                    y_lim = [0,110]
                    name_area = 'A-A'
                elif before_area == 'B-before' and after_area == 'A-after':
                    y_lim = [0, 70]
                    name_area = 'B-A'
                elif before_area == 'C-before' and after_area == 'A-after':
                    y_lim = [0, 20]
                    name_area = 'C-A'
            if 0 in day_of_week:
                name_day_week = 'Weekdays'
            else:
                name_day_week = 'Weekends'
            if duration_or_volume == 'duration':
                x_lim = [hour_lim[0] - 0.5, hour_lim[1] +1]
            else:
                x_lim = [hour_lim[0] - 0.5, hour_lim[1] + 0.5]

            plt.ylim(y_lim)
            plt.xlim(x_lim)
            x_text = (hour_lim[1] - hour_lim[0]) * 0.05 + hour_lim[0]
            y_text = (y_lim[1] - y_lim[0]) * 0.09 + y_lim[0]
            plt.text(x_text, y_text, name_area + '; ' +  name_day_week, fontsize = font_size * 1.1,
                     bbox=dict(facecolor='red', alpha=0.08))

            if save_fig == 0:
                plt.show()
            else:
                day_of_week_str = [str(i) for i in day_of_week]
                name_tail = '_'.join(day_of_week_str)
                if duration_or_volume == 'duration':
                    plt.savefig('img/Time_of_day_duration_' + name_tail + '_' + before_area + '_' + after_area + '.png', dpi=200)
                else:
                    plt.savefig('img/Time_of_day_' + name_tail + '_' + before_area + '_' + after_area + '.png', dpi=200)

def bins_subarea(data, day_of_week_list, subarea_list, duration_or_volume, save_fig=0):
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
            if duration_or_volume == 'duration':
                data_area = data_area.groupby(['PS_year', 'PS_month', 'PS_day', 'day_of_week'])['Parking_duration'].mean()
            else:
                data_area = data_area.groupby(['PS_year','PS_month','PS_day','day_of_week'])['id'].count()
            data_area = data_area.reset_index(drop=False)
            data_area = data_area.loc[data_area['day_of_week'].isin(day_list)]
            if duration_or_volume == 'duration':
                data_area = data_area.rename(columns={'Parking_duration': 'parking_vol'})
            else:
                data_area = data_area.rename(columns = {'id':'parking_vol'})
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
        if duration_or_volume == 'duration':
            ax.set_ylabel('Parking duration (min)', fontsize=16)
            y_lim = [0,80]
        else:
            ax.set_ylabel('Parking volume (# veh/day)', fontsize=16)
            y_lim = [0, 1100]
        plt.ylim(y_lim[0], y_lim[1])
        if duration_or_volume == 'duration':
            for loc, y1loc, y2loc, area, in zip(ind, before_mean, after_mean, subarea_list):
                delta = 0.04
                x1 = loc - delta
                x2 = loc + width - delta
                y1 = 0.5 * y1loc
                y2 = 0.5 * y2loc
                plt.text(x1, y1, str(int(round(y1loc))), fontsize=font_size)
                plt.text(x2, y2, str(int(round(y2loc))), fontsize=font_size)
                reduction_per = (y2loc - y1loc) / y1loc
                star = star_plot(p_value_ks[area])
                plt.text(x2 - delta * 1.7, y2 - 0.05*y_lim[1], '(' + str(round(reduction_per * 100, 1)) + '%' + ')',
                         fontsize=font_size)
                plt.text(x2 - delta * 0.5, y2 - 0.1*y_lim[1], star, fontsize=font_size - 5)
        else:
            for loc,y1loc,y2loc,area, in zip(ind,before_mean,after_mean,subarea_list):
                delta = 0.06
                x1 = loc-delta
                x2 = loc + width-delta
                y1 = 0.5*y1loc
                y2 = 0.5*y2loc
                plt.text(x1,y1,str(int(round(y1loc))), fontsize = font_size)
                plt.text(x2,y2,str(int(round(y2loc))), fontsize = font_size)
                reduction_per = (y2loc-y1loc)/y1loc
                star = star_plot(p_value_ks[area])
                plt.text(x2-delta*0.9, y2 - 65, '('+str(round(reduction_per*100,1))  + '%'+ ')'  , fontsize=font_size)
                plt.text(x2 - delta * 0.5, y2 - 120, star   , fontsize=font_size-5)
        #
        plt.yticks(fontsize=font_size)
        #

        ax.set_xticks(ind+0.5*width)
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
            if duration_or_volume == 'duration':
                plt.savefig('img/Bins_of_parking_dur_' + name_tail + '.png', dpi=200)
            else:
                plt.savefig('img/Bins_of_parking_vol_' + name_tail + '.png', dpi=200)

def temporal_analysis_ACF(data, save_fig=0):
    data_vol = data.groupby(['PS_year','PS_month','PS_day','day_of_week','before_area','after_area'])['id'].count()
    data_vol = data_vol.reset_index(drop=False)
    data_vol = data_vol.rename(columns = {'id':'parking_vol'})

    data_vol_city = data_vol.groupby(['PS_year','PS_month','PS_day','day_of_week'])['parking_vol'].sum()
    data_vol_city = data_vol_city.reset_index(drop=False)

    data_vol_city['day_id'] = range(0,len(data_vol_city))
    threshold = data_vol_city.loc[(data_vol_city['PS_year']==2014)&(data_vol_city['PS_month']==12)&(data_vol_city['PS_day']==30), 'day_id'].values[0]

    x_ticks = data_vol_city.loc[(data_vol_city['PS_day']==1)&(data_vol_city['PS_month'].isin([1,3,5,7,9,11])),'day_id']
    x_ticks = x_ticks.append(data_vol_city.loc[(data_vol_city['PS_day']==31)&(data_vol_city['PS_month']==8)&(data_vol_city['PS_year']==2015),'day_id'])

    new_ticks = from_dayid_to_date(x_ticks,data_vol_city)

    data_vol_city = delete_holidays(data_vol_city)

    x = np.array(data_vol_city['day_id'])
    y = np.array(data_vol_city['parking_vol'])


    y_before = np.array(data_vol_city.loc[(data_vol_city['PS_year']==2014),'parking_vol'])
    y_after = np.array(data_vol_city.loc[(data_vol_city['PS_year'] == 2015), 'parking_vol'])
    y_before_weekdays = np.array(data_vol_city.loc[(data_vol_city['PS_year']==2014)&
                                                   (data_vol_city['day_of_week'].isin([0,1,2,3,4])),'parking_vol'])
    y_after_weekdays = np.array(data_vol_city.loc[(data_vol_city['PS_year'] == 2015)&
                                                  (data_vol_city['day_of_week'].isin([0,1,2,3,4])), 'parking_vol'])
    # y_after_weekends = np.array(data_vol_city.loc[(data_vol_city['PS_year'] == 2015)&
    #                                               (data_vol_city['day_of_week'].isin([5,6])), 'parking_vol'])
    # sm.graphics.tsa.plot_pacf(y_before)
    y_plot = [y_before_weekdays,y_after_weekdays]

    ####Edit figures legends,labels.

    for i in range(2):
        fig, ax = plt.subplots(figsize=(9, 5))
        font_size = 16
        max_lag = 18
        sm.graphics.tsa.plot_acf(y_plot[i],ax=ax, lags= max_lag)
        inter = 1
        x_ticks_new = list(range(0,max_lag + inter,inter))
        x_lim_new = [x_ticks_new[0],x_ticks_new[-1]]
        plt.xlabel('Lag', fontsize=font_size)
        plt.ylabel('Autocorrelation', fontsize=font_size)
        plt.yticks(fontsize=font_size)
        plt.xticks(x_ticks_new, fontsize=font_size)
        plt.xlim(x_lim_new[0]-1, x_lim_new[1]+1)
        ax.set_title('')
        plt.tight_layout()

        if save_fig == 0:
            plt.show()
        else:
            if i == 1:
                name_tail = 'before'
            else:
                name_tail = 'after'
            plt.savefig('img/ACF_' + name_tail + '.png', dpi=200)


def temporal_analysis(data, year_month_list, duration_or_volume, save_fig = 0):
    if duration_or_volume == 'duration':
        data_vol = data.groupby(['PS_year','PS_month','PS_day','day_of_week','before_area','after_area'])['Parking_duration'].mean()
        data_vol = data_vol.reset_index(drop=False)
        data_vol = data_vol.rename(columns = {'Parking_duration':'parking_vol'})
        data_vol_city = data_vol.groupby(['PS_year', 'PS_month', 'PS_day', 'day_of_week'])['parking_vol'].mean()
    else:
        data_vol = data.groupby(['PS_year','PS_month','PS_day','day_of_week','before_area','after_area'])['id'].count()
        data_vol = data_vol.reset_index(drop=False)
        data_vol = data_vol.rename(columns = {'id':'parking_vol'})
        data_vol_city = data_vol.groupby(['PS_year','PS_month','PS_day','day_of_week'])['parking_vol'].sum()

    data_vol_city = data_vol_city.reset_index(drop=False)

    data_vol_city['day_id'] = range(0,len(data_vol_city))
    threshold = data_vol_city.loc[(data_vol_city['PS_year']==2014)&(data_vol_city['PS_month']==12)&(data_vol_city['PS_day']==30), 'day_id'].values[0]

    data_vol_city = delete_holidays(data_vol_city)

    colors = sns.color_palette("Paired")

    def draw_line_curve(data_vol_city, save_fig):
        x_ticks = data_vol_city.loc[
            (data_vol_city['PS_day'] == 1) & (data_vol_city['PS_month'].isin([1, 3, 5, 7, 9, 11])), 'day_id']
        x_ticks = x_ticks.append(data_vol_city.loc[
                                     (data_vol_city['PS_day'] == 31) & (data_vol_city['PS_month'] == 8) & (
                                                 data_vol_city['PS_year'] == 2015), 'day_id'])

        new_ticks = from_dayid_to_date(x_ticks, data_vol_city)

        x = np.array(data_vol_city['day_id'])
        y = np.array(data_vol_city['parking_vol'])

        fig, ax = plt.subplots(figsize=(10, 6))
        nbins = 24
        n, _ = np.histogram(x, bins=nbins)
        sy, _ = np.histogram(x, bins=nbins, weights=y)
        sy2, _ = np.histogram(x, bins=nbins, weights=y * y)
        mean = sy / n
        std = np.sqrt(sy2 / n - mean * mean)

        plt.axvline(x=threshold, color='b')

        font_size = 16
        plt.plot(x, y, 'ko',markersize = 5)
        plt.errorbar((_[1:] + _[:-1]) / 2, mean, yerr=std, fmt='r-')
        x_lim = [x_ticks.iloc[0], x_ticks.iloc[-1]]
        plt.xlabel('Time', fontsize=font_size)
        if duration_or_volume == 'duration':
            plt.ylabel('Parking duration (min)', fontsize=font_size)
        else:
            plt.ylabel('Parking volume (# veh/day)', fontsize=font_size)
        plt.yticks(fontsize=font_size)
        plt.xticks(x_ticks, new_ticks, fontsize=font_size)
        plt.xlim(x_lim[0]-5, x_lim[1]+5)
        plt.tight_layout()
        if save_fig == 0:
            plt.show()
        else:
            if duration_or_volume == 'duration':
                plt.savefig('img/Temporal_analysis_duration.jpg', dpi=300)
            else:
                plt.savefig('img/Temporal_analysis_volume.jpg', dpi=300)


    def draw_box(data_vol_city, year_month_list, save_fig):

        fig, ax1 = plt.subplots(figsize=(10, 6))
        font_size = 16
        #data_vol_city = data_vol_city.loc[data_vol_city['day_of_week'].isin([0,1,2,3,4])]
        data_to_plot = [data_vol_city.loc[(data_vol_city['PS_year'] == key[0])&
                                          (data_vol_city['PS_month'] == key[1]),'parking_vol'] for key in year_month_list]
        bp = ax1.boxplot(data_to_plot, patch_artist=False, showfliers=False,whiskerprops = {'alpha':1}, showcaps = True)
        #####Draw true beta
        data_vol_city_month = data_vol_city.groupby(['PS_year','PS_month']).mean()
        data_vol_city_month = data_vol_city_month.reset_index(drop=False)
        before_x = [1,2,3,4]
        before_y = data_vol_city_month.iloc[0:4]['parking_vol']

        after_x = [5,6,7,8,9,10,11,12]
        after_y = data_vol_city_month.iloc[4:12]['parking_vol']

        plt.plot(before_x,before_y, '--s', markersize = 5 , color = 'r')
        plt.plot(after_x, after_y, '--s', markersize=5, color='r', label = 'Mean')

        mean_before = before_y.mean()
        plt.plot([4.5,13], [mean_before,mean_before], '-', markersize=5, color=colors[0], linewidth = 3, label = 'Mean of before')
        #
        for k in range(len(after_x)):
            if k == 0:
                plt.plot([after_x[k], after_x[k]], [mean_before, after_y.iloc[k]], '-',linewidth = 3, color=colors[3], alpha = 0.5, label='Avg reduction')
            else:
                plt.plot([after_x[k], after_x[k]], [mean_before, after_y.iloc[k]], '-', linewidth=3, color=colors[3],alpha = 0.5)
        #######
        if duration_or_volume == 'duration':
            ax1.set_ylabel('Parking duration (min)', fontsize=font_size)
        else:
            ax1.set_ylabel('Parking volume', fontsize=font_size)
        ax1.set_xlabel('Month', fontsize=font_size)
        ax1.set_xticklabels([str(key[0]).replace('20','') + '/'+ str(key[1]) for key in year_month_list])

        ax1.tick_params(axis='y', labelsize=font_size)
        ax1.tick_params(axis='x', labelsize=font_size)
        plt.axvline(x=4.5, color='b')
        ## change outline color, fill color and linewidth of the boxes
        # for box in bp['boxes']:
        #     # change outline color
        #     box.set(color='#7570b3', linewidth=2)
        #     # change fill color
        #     box.set(facecolor='#1b9e77')
        #
        # ## change color and linewidth of the whiskers
        # for whisker in bp['whiskers']:
        #     whisker.set(color='#7570b3', linewidth=2)
        #
        # ## change color and linewidth of the caps
        # for cap in bp['caps']:
        #     cap.set(color='#7570b3', linewidth=2)
        #
        # ## change color and linewidth of the medians
        # for median in bp['medians']:
        #     median.set(color='#b2df8a', linewidth=2)
        #
        # ## change the style of fliers and their fill
        # for flier in bp['fliers']:
        #     flier.set(marker='o', color='#e7298a', alpha=0.5)
        plt.legend(fontsize=15)
        plt.xlim([0.5,12.5])
        if duration_or_volume == 'duration':
            plt.ylim([0, 100])
        else:
            plt.ylim([0, 2000])

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        if save_fig == 0:
            plt.show()
        else:
            if duration_or_volume == 'duration':
                plt.savefig('img/Temporal_boxplot_duration.jpg', dpi=300)
            else:
                plt.savefig('img/Temporal_boxplot_volume.jpg', dpi=300)

    draw_box(data_vol_city, year_month_list, save_fig)




def delete_holidays(data):
    data['datetime'] = data[['PS_year','PS_month','PS_day']].apply(lambda x: datetime.date(x[0],x[1],x[2]),axis = 1)
    data['is_holidays'] = data['datetime'].apply(lambda x: is_holiday(x))
    data.loc[data['day_of_week'] == 5, 'is_holidays'] = False
    data.loc[data['day_of_week'] == 6, 'is_holidays'] = False
    # Luner March 3 (san yue san)
    # 2015 04 21 04 20
    data.loc[data['datetime']== datetime.date(2015,4,20),'is_holidays'] = True
    data.loc[data['datetime']== datetime.date(2015,4,21),'is_holidays'] = True


    # Spring festival impact
    data.loc[data['datetime']== datetime.date(2015,2,21),'is_holidays'] = True
    data.loc[data['datetime']== datetime.date(2015,2,22),'is_holidays'] = True

    data.loc[data['datetime']== datetime.date(2015,2,17),'is_holidays'] = True
    data.loc[data['datetime']== datetime.date(2015,2,25),'is_holidays'] = True
    data.loc[data['datetime']== datetime.date(2015,2,26),'is_holidays'] = True
    return data.loc[~data['is_holidays']]
if __name__ == '__main__':
    RDD = False
    TIME_OF_DAY = True
    ###
    ACF = False
    ############data analysis
    meter_data_info = pd.read_csv('data/meter_data_info.csv')
    fix_before_zero = True
    save_fig = 1
    ########################################### RD city
    if RDD:
        duration_or_volume_list = ['duration'] #duration, volume
        for duration_or_volume in duration_or_volume_list:
            RD_design_city_level(meter_data_info,duration_or_volume, save_fig=save_fig, fix_before_zero = fix_before_zero)
            # # ############################# RD weekdays weekends
            day_of_week_list = [[0,1,2,3,4],[5,6]]
            RD_design_city_level_weekdays_weekends(meter_data_info,duration_or_volume, day_of_week_list, save_fig = save_fig, fix_before_zero = fix_before_zero)

            #################################################### RD Subarea level
            subarea_list=[('A-before','A-after'),('B-before','A-after')] # ('C-before','A-after')
            RD_design_subarea_level(meter_data_info, duration_or_volume, subarea_list, save_fig=save_fig)


    if TIME_OF_DAY:
        # ########## time of day distribution
        duration_or_volume_list = ['duration','volume']  # duration, volume

        for duration_or_volume in duration_or_volume_list:
            day_of_week_list = [[0, 1, 2, 3, 4], [5, 6]]
            month_dict= {'Before':[2014,[9,10,11,12]], 'After':[2015,[1,2,3,4,5,6,7,8]]}
            time_of_day_distribution(meter_data_info,day_of_week_list, month_dict, duration_or_volume, save_fig = save_fig)
            #
            # ################################ time of day distribution
            subarea_list = [('A-before', 'A-after'), ('B-before', 'A-after')]
            day_of_week_list = [[0, 1, 2, 3, 4], [5, 6]]
            month_dict = {'Before': [2014, [9, 10, 11, 12]], 'After': [2015, [1, 2, 3, 4, 5, 6, 7, 8]]}
            time_of_day_distribution_subarea(meter_data_info, subarea_list, day_of_week_list, month_dict, duration_or_volume, save_fig =save_fig)

            ### bins
            subarea_list = [('A-before', 'A-after'), ('B-before', 'A-after')]
            day_of_week_list = [[0, 1, 2, 3, 4], [5, 6]]
            bins_subarea(meter_data_info, day_of_week_list, subarea_list,duration_or_volume, save_fig=save_fig)

            # #########################Temporal analysis
            year_month_list = [(2014, 9),(2014, 10),(2014, 11),(2014, 12),
                               (2015, 1),(2015, 2),(2015, 3),(2015, 4),
                               (2015, 5),(2015, 6),(2015, 7),(2015, 8)]
            temporal_analysis(meter_data_info, year_month_list, duration_or_volume, save_fig=save_fig)
            if ACF:
                temporal_analysis_ACF(meter_data_info, save_fig=save_fig)