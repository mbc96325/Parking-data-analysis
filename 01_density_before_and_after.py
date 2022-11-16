'''
Plot before and after data analysis curve
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model

#pre_process
#data_2014 = pd.read_excel('data/data_2014_format.xlsx')
#data_2015 = pd.read_excel('data/data_2015_format.xlsx')
#data_201415 = pd.read_excel('data/data_combined1415_format.xlsx')
#data_2014.to_csv('data/data_2014.csv',index=False)
#data_2015.to_csv('data/data_2015.csv',index=False)
#data_201415.to_csv('data/data_201415.csv',index=False)
#-----------------------
data_2014 = pd.read_csv('data/data_2014_new.csv')
data_2015 = pd.read_csv('data/data_2015_new.csv')
#data_201415 = pd.read_csv('data/data_201415.csv')
total_2014 = len(data_2014)
total_2015 = len(data_2015)
print ('num 2014', total_2014)
print ('num 2015', total_2015)
labels = [2014,2015]
colors = ["black", "gray"]

def plot_gender(save_fig=0):
    density_2014 = []
    density_2015 = []
    labels_list = ('Female','Male')
    columns = pd.unique(data_2015['MALE'].loc[data_2015['MALE']!=-1])
    columns = sorted(columns)
    #print (columns)
    total_2014 = len(data_2014['MALE'].loc[data_2014['MALE']!=-1])
    total_2015 = len(data_2015['MALE'].loc[data_2015['MALE']!=-1])

    for key in columns:
        num = len(data_2014['MALE'].loc[data_2014['MALE']==key])
        density_2014.append(num/total_2014)        
        num = len(data_2015['MALE'].loc[data_2015['MALE']==key])
        density_2015.append(num/total_2015)
    N = len(columns)
    ind = np.arange(N)  # the x locations for the groups
    width = 0.3      # the width of the bars
    fig, ax = plt.subplots(figsize=(8, 6))
    rects1 = ax.bar(ind, density_2014, width, color=colors[0])
    rects2 = ax.bar(ind+width, density_2015, width, color=colors[1])
    print ('----------------')
    print('gender', columns)        
    print ('2014 gender', density_2014)
    print ('2015 gender', density_2015)       
    plt.yticks(fontsize=16)
    #plt.ylim(0.35,0.46)
    #ax.set_yticklabels([str(i) + '%' for i in range(40, 61, 10)])    
    ax.set_ylabel('Density',fontsize=16)
    ax.set_xticks(ind+0.5*width)
    ax.set_xticklabels(labels_list,fontsize=16)
    ax.set_xlabel('Gender',fontsize=16)
    ax.legend( (rects1[0], rects2[0]) , labels, fontsize=16, loc='upper right')
    plt.tight_layout()
    if save_fig == 0:
        plt.show()
    else:
        plt.savefig('img/Income_2014_2015.png', dpi=300)
def plot_income(save_fig=0):
    density_2014 = []
    density_2015 = []
    labels_list = ('<2.5','2.5~5.0','5.0~7.5','7.5~10','>10')
    columns = pd.unique(data_2015['INCOME'].loc[data_2015['INCOME']!=-1])
    columns = sorted(columns)
    #print (columns)
    total_2014 = len(data_2014['INCOME'].loc[data_2014['INCOME']!=-1])
    total_2015 = len(data_2015['INCOME'].loc[data_2015['INCOME']!=-1])

    for key in columns:
        num = len(data_2014['INCOME'].loc[data_2014['INCOME']==key])
        density_2014.append(num/total_2014)        
        num = len(data_2015['INCOME'].loc[data_2015['INCOME']==key])
        density_2015.append(num/total_2015)
    N = len(columns)
    ind = np.arange(N)  # the x locations for the groups
    width = 0.3      # the width of the bars
    fig, ax = plt.subplots(figsize=(8, 6))
    rects1 = ax.bar(ind, density_2014, width, color=colors[0])
    rects2 = ax.bar(ind+width, density_2015, width, color=colors[1])
    print ('----------------')
    print('income', columns)       
    print ('2014 income', density_2014)
    print ('2015 income', density_2015)    
    plt.yticks(fontsize=16)
    #plt.ylim(0.35,0.46)
    #ax.set_yticklabels([str(i) + '%' for i in range(40, 61, 10)])    
    ax.set_ylabel('Density',fontsize=16)
    ax.set_xticks(ind+0.5*width)
    ax.set_xticklabels(labels_list,fontsize=16)
    ax.set_xlabel('Individual Monthly Income ('+'RMB'+' 1000)',fontsize=16)
    ax.legend( (rects1[0], rects2[0]) , labels, fontsize=16, loc='upper right')
    plt.tight_layout()
    if save_fig == 0:
        plt.show()
    else:
        plt.savefig('img/Income_2014_2015.png', dpi=300)
                 
             
def plot_age(save_fig=0):
    density_2014 = []
    density_2015 = []
    labels_list = ('18~30','31~40','41~50','51~60','>60')
    data_2015.loc[data_2015['AGE']==0,'AGE'] = 1
    data_2014.loc[data_2014['AGE']==0,'AGE'] = 1
    columns = pd.unique(data_2015['AGE'].loc[data_2015['AGE']!=-1])
    columns = sorted(columns)
    #print (columns)
    total_2014 = len(data_2014['AGE'].loc[data_2014['AGE']!=-1])
    total_2015 = len(data_2015['AGE'].loc[data_2015['AGE']!=-1])

    
    for key in columns:
        num = len(data_2014['AGE'].loc[data_2014['AGE']==key])
        density_2014.append(num/total_2014)        
        num = len(data_2015['AGE'].loc[data_2015['AGE']==key])
        density_2015.append(num/total_2015)
    N = len(columns)
    ind = np.arange(N)  # the x locations for the groups
    width = 0.3      # the width of the bars
    fig, ax = plt.subplots(figsize=(8, 6))
    rects1 = ax.bar(ind, density_2014, width, color=colors[0])
    rects2 = ax.bar(ind+width, density_2015, width, color=colors[1])
    print ('----------------')
    print('age', columns)    
    print ('2014 age', density_2014)
    print ('2015 age', density_2015)
    plt.yticks(fontsize=16)
    #plt.ylim(0.35,0.46)
    #ax.set_yticklabels([str(i) + '%' for i in range(40, 61, 10)])    
    ax.set_ylabel('Density',fontsize=16)
    ax.set_xticks(ind+0.5*width)
    ax.set_xticklabels(labels_list,fontsize=16)
    ax.set_xlabel('Age',fontsize=16)
    ax.legend( (rects1[0], rects2[0]) , labels, fontsize=16, loc='upper right')
    plt.tight_layout()
    if save_fig == 0:
        plt.show()
    else:
        plt.savefig('img/Age_2014_2015.png', dpi=300)             

def area_number():
    A_2014 = len(data_2014.loc[data_2014['AREA']==1])
    print ('2014 A', A_2014)
    B_2014 = len(data_2014.loc[data_2014['AREA']==2])
    print ('2014 B', B_2014)
    C_2014 = total_2014 - A_2014 - B_2014
    print ('2014 C', C_2014)
    #--------------------------------------------
    A_2015 = len(data_2015.loc[data_2015['AREA']==1])
    print ('2015 A', A_2015)
    B_2015 = len(data_2015.loc[data_2015['AREA']==2])
    print ('2015 B', B_2015)
    C_2015 = total_2015 - A_2015 - B_2015
    print ('2015 C', C_2015) 
    
def proportions():
    density_2014 = []
    density_2015 = []
    columns = pd.unique(data_2015['HAVE_PARKING'].loc[data_2015['HAVE_PARKING']!=-1])
    columns = sorted(columns)
    print (columns)
    total_2014 = len(data_2014['HAVE_PARKING'].loc[data_2014['HAVE_PARKING']!=-1])
    total_2015 = len(data_2015['HAVE_PARKING'].loc[data_2015['HAVE_PARKING']!=-1])

    
    for key in columns:
        num = len(data_2014['HAVE_PARKING'].loc[data_2014['HAVE_PARKING']==key])
        density_2014.append(num/total_2014)        
        num = len(data_2015['HAVE_PARKING'].loc[data_2015['HAVE_PARKING']==key])
        density_2015.append(num/total_2015) 
    print ('----------------')
    print('parking', columns)
    print ('2014 parking', density_2014)
    print ('2015 parking', density_2015)   

def parking_related_variables(data_2014,if_2014):
    data_copy_2014 = data_2014.copy()
    print ('WORKING:', data_copy_2014['WORKING'].mean(), data_copy_2014['WORKING'].std())
    print ('RECREATION:', data_copy_2014['RECREATION'].mean(), data_copy_2014['RECREATION'].std())
    data_copy_2014['OTHER'] = 1 - data_copy_2014['WORKING'] - data_copy_2014['RECREATION']
    print ('OTHER:', data_copy_2014['OTHER'].mean(), data_copy_2014['OTHER'].std())
    #---
    print ('SELF_PAY:', data_copy_2014['SELF_PAY'].mean(), data_copy_2014['SELF_PAY'].std())
    data_copy_2014['OTHER_PAY'] = 1 - data_copy_2014['SELF_PAY'] 
    print ('OTHER_PAY:', data_copy_2014['OTHER_PAY'].mean(), data_copy_2014['OTHER_PAY'].std())    
    #---
    print ('ILL_NEVER:', data_copy_2014['ILL_NEVER'].mean(), data_copy_2014['ILL_NEVER'].std())
    data_copy_2014['ILL_OFTEN'] = 0
    data_copy_2014.loc[data_copy_2014['ILLEGAL_FRE']==2,'ILL_OFTEN'] = 1
    print ('ILL_OFTEN:', data_copy_2014['ILL_OFTEN'].mean(), data_copy_2014['ILL_OFTEN'].std())
    data_copy_2014['ILL_SOMETIMES'] = 1 - data_copy_2014['ILL_OFTEN'] - data_copy_2014['ILL_NEVER']  
    print ('ILL_SOMETIMES:', data_copy_2014['ILL_SOMETIMES'].mean(), data_copy_2014['ILL_SOMETIMES'].std())        
    #-----
    if if_2014:
        #data_copy_2014.loc[data_copy_2014['EP_TIME']<-400]
        print ('EX_PARKING_TIME:', data_copy_2014['EP_TIME'].mean(), data_copy_2014['EP_TIME'].std())
    else:
        print ('EX_PARKING_TIME:', data_copy_2014['EP_TIME'].mean(), data_copy_2014['EP_TIME'].std())
    #---
    data_copy_2014.loc[(data_copy_2014['SUN_WEB']+data_copy_2014['RAIN_WEB'])>1,'SUN_WEB'] = 0
    data_copy_2014.loc[(data_copy_2014['CLOUD_WEB']+data_copy_2014['RAIN_WEB'])>1,'CLOUD_WEB'] = 0
    data_copy_2014.loc[(data_copy_2014['SUN_WEB']+data_copy_2014['CLOUD_WEB'])>1,'CLOUD_WEB'] = 0
    
    print ('SUN_WEB:', data_copy_2014['SUN_WEB'].mean(), data_copy_2014['SUN_WEB'].std())
    print ('CLOUD_WEB:', data_copy_2014['CLOUD_WEB'].mean(), data_copy_2014['CLOUD_WEB'].std())
    print ('RAIN_WEB:', data_copy_2014['RAIN_WEB'].mean(), data_copy_2014['RAIN_WEB'].std())   
    #-----    
    data_copy_2014['WEEKDAY'] = 0
    data_copy_2014['WEEKEND'] = 0
    data_copy_2014.loc[data_copy_2014['DAY_WEEK']<=5,'WEEKDAY'] = 1
    data_copy_2014.loc[data_copy_2014['DAY_WEEK']>5,'WEEKEND'] = 1
    print ('WEEKDAY:', data_copy_2014['WEEKDAY'].mean(), data_copy_2014['WEEKDAY'].std())
    print ('WEEKEND:', data_copy_2014['WEEKEND'].mean(), data_copy_2014['WEEKEND'].std())
    print ('TEMP:', data_copy_2014['TEMP'].mean(), data_copy_2014['TEMP'].std())
    
def expected_parking_time(data_2014):
    #Density Plot and Histogram of all arrival delays
    #data_2014 = data_2014.loc[data_2014['EP_TIME']<1000]
    sns.distplot(data_2014['EP_TIME'], hist=False, kde=True, color = colors[0], 
                 hist_kws={'edgecolor':'black'},
                 kde_kws={'linewidth': 4})   
    sns.distplot(data_2015['EP_TIME'], hist=False, kde=True, color = colors[1], 
                 hist_kws={'edgecolor':'black'},
                 kde_kws={'linewidth': 4})       
    plt.show()

def parking_characteristic(save_fig):
    # AP_FEE AP_TIME CHARGE
    #colors = ["#3366cc", "#dc3912"]
    #sns.set(font_scale=1.5)
    #sns.set_style("white", {"legend.frameon": True})
    #fig, ax = plt.subplots(figsize=(10, 5))    

    
    #data_2014_copy = data_2014.loc[data_2014['AP_TIME']<=400]
    #sns.kdeplot(data_2014_copy['AP_TIME'], ax=ax, shade=True, color=colors[0], label=labels[0],bw=0.22)
    #sns.kdeplot(data_2015['AP_TIME'], ax=ax, shade=True, color=colors[1], label=labels[1],bw=0.22)  
    
    #meda = data_2014_copy['AP_TIME'].mean()
    #plt.axvline(meda, color=colors[0], linestyle='dashed', linewidth=2)
    #plt.text(meda + 6, 0.016, 'Mean = {}'.format(round(meda, 1)),
        #horizontalalignment='left', verticalalignment='center',
        #fontsize=15, color=colors[0])    
    
    #medb = data_2015['AP_TIME'].mean()
    #plt.axvline(medb, color=colors[1], linestyle='dashed', linewidth=2)
    #plt.text(medb + 6, 0.016, 'Mean = {}'.format(round(medb, 1)),
        #horizontalalignment='left', verticalalignment='center',
        #fontsize=15, color=colors[1])   
    #plt.xlim(0, 400)
    #new_ticks = list(range(0,400,60))
    ##new_ticks.append(30)
    #plt.xticks(new_ticks)   
    #new_ticks2 = list(np.arange(0,0.022,0.004))
    #plt.ylim(0, 0.020)
    ##new_ticks.append(30)
    #plt.yticks(new_ticks2)      
    #plt.xlabel('Actual Parking Durtation (min)', fontsize=16)
    #plt.ylabel('Density', fontsize=16)     
    #plt.tight_layout()
    #if save_fig == 0:
        #plt.show()
    #else:
        #plt.savefig('img/actual_duration_2014_2015.png', dpi=300)    
    
    ##---------------------------------------------
    #sns.set(font_scale=1.5)
    #sns.set_style("white", {"legend.frameon": True})    
    #fig, ax = plt.subplots(figsize=(10, 5))    
    
    #sns.kdeplot(data_2014['AP_FEE'], ax=ax, shade=True, color=colors[0], label=labels[0])
    #sns.kdeplot(data_2015['AP_FEE'], ax=ax, shade=True, color=colors[1], label=labels[1])  
    
    #meda = data_2014['AP_FEE'].mean()
    #plt.axvline(meda, color=colors[0], linestyle='dashed', linewidth=2)
    #plt.text(meda-0.5, 0.165, 'Mean = {}'.format(round(meda, 1)),
        #horizontalalignment='right', verticalalignment='center',
        #fontsize=15, color=colors[0])    
    
    #medb = data_2015['AP_FEE'].mean()
    #plt.axvline(medb, color=colors[1], linestyle='dashed', linewidth=2)
    #plt.text(medb+0.5, 0.165, 'Mean = {}'.format(round(medb, 1)),
        #horizontalalignment='left', verticalalignment='center',
        #fontsize=15, color=colors[1])   
    #plt.xlim(0, 30)
    #new_ticks2 = list(np.arange(0,0.18,0.03))
    #plt.ylim(0, 0.18)
    ##new_ticks.append(30)
    #plt.yticks(new_ticks2)      
    ##new_ticks = list(range(0,400,60))
    ##new_ticks.append(30)
    ##plt.xticks(new_ticks)      
    #plt.xlabel('Actual Total Parking Fee (RMB)', fontsize=16)
    #plt.ylabel('Density', fontsize=16)    
    #plt.tight_layout()
    
    #if save_fig == 0:
        #plt.show()
    #else:
        #plt.savefig('img/actual_price_2014_2015.png', dpi=300)        

    #--------------------------------------
    colors = ["black", "gray"]
    font_size = 20
    density_2014 = []
    density_2015 = []
    labels_list = ('Inexpensive','Moderate', 'Expensive')
    columns = pd.unique(data_2015['CHARGE'].loc[data_2015['CHARGE']!=-1])
    columns = sorted(columns)
    #print (columns)
    total_2014 = len(data_2014['CHARGE'].loc[data_2014['CHARGE']!=-1])
    total_2015 = len(data_2015['CHARGE'].loc[data_2015['CHARGE']!=-1])

    for key in columns:
        num = len(data_2014['CHARGE'].loc[data_2014['CHARGE']==key])
        density_2014.append(num/total_2014)        
        num = len(data_2015['CHARGE'].loc[data_2015['CHARGE']==key])
        density_2015.append(num/total_2015)
    N = len(columns)
    ind = np.arange(N)  # the x locations for the groups
    width = 0.27      # the width of the bars
    fig, ax = plt.subplots(figsize=(8, 6))
    rects1 = ax.bar(ind, density_2014, width, color=colors[0])
    rects2 = ax.bar(ind+width, density_2015, width, color=colors[1])
    print ('----------------')
    print('CHARGE', columns)       
    print ('2014 CHARGE', density_2014)
    print ('mean', data_2014['CHARGE'].mean(), 'std', data_2014['CHARGE'].std())
    print ('2015 CHARGE', density_2015)    
    print ('mean', data_2015['CHARGE'].mean(), 'std', data_2015['CHARGE'].std())
    plt.yticks(fontsize=font_size)
    #plt.ylim(0.35,0.46)
    #ax.set_yticklabels([str(i) + '%' for i in range(40, 61, 10)])    
    ax.set_ylabel('Density',fontsize=font_size)
    ax.set_xticks(ind+0.5*width)
    ax.set_xticklabels(labels_list,fontsize=font_size)
    ax.set_xlabel('Perceived Parking Price',fontsize=font_size)
    ax.legend( (rects1[0], rects2[0]) , labels, fontsize=font_size, loc='upper right')
    plt.tight_layout()
    if save_fig == 0:
        plt.show()
    else:
        plt.savefig('img/Perceived_price_2014_2015.png', dpi=300)   
def scatter_duration():
    #data_2014_copy =  data_2014.loc[data_2014['AP_TIME']<=400]
    #data_2015_copy =  data_2015.loc[data_2015['AP_TIME']<=400]
    data_2014_copy = data_2014.copy()
    data_2015_copy = data_2015.copy()
    #-------------
    font_size = 16

    plt.subplots(figsize=(6.5, 6))

    plt.scatter(data_2014_copy['AP_TIME'],data_2014_copy['EP_TIME'],s=10)
    plt.plot(np.array([0,data_2014_copy['AP_TIME'].max()*1.5]),np.array([0,data_2014_copy['AP_TIME'].max()*1.5]),'k--')
    #----
    regr = linear_model.LinearRegression()
    train_x = np.array([data_2014_copy['AP_TIME']]).reshape((-1, 1))
    train_y = np.array([data_2014_copy['EP_TIME']]).reshape((-1, 1))
    regr.fit(train_x,train_y)
    predic_x = np.array([0,data_2014_copy['AP_TIME'].max()*1.5]).reshape((-1, 1))
    diabetes_y_pred = regr.predict(predic_x)
    plt.plot(predic_x,diabetes_y_pred,'r--')
    #----
    plt.yticks(fontsize=font_size)
    plt.xticks(fontsize=font_size)
    plt.ylim(0,410)
    plt.xlim(0,410)
    plt.xlabel('Actual Parking Duration (min)',fontsize=font_size)
    plt.ylabel('Expected Parking Duration (min)',fontsize=font_size)
    plt.tight_layout()
    plt.show()    
    #-------------
    font_size = 16

    plt.subplots(figsize=(6.5, 6))

    plt.scatter(data_2015_copy['AP_TIME'],data_2015_copy['EP_TIME'],s=10)
    plt.plot(np.array([0,data_2015_copy['AP_TIME'].max()*1.5]),np.array([0,data_2015_copy['AP_TIME'].max()*1.5]),'k--')
    #----
    regr = linear_model.LinearRegression()
    train_x = np.array([data_2015_copy['AP_TIME']]).reshape((-1, 1))
    train_y = np.array([data_2015_copy['EP_TIME']]).reshape((-1, 1))
    regr.fit(train_x,train_y)
    predic_x = np.array([0,data_2015_copy['AP_TIME'].max()*1.5]).reshape((-1, 1))
    diabetes_y_pred = regr.predict(predic_x)
    plt.plot(predic_x,diabetes_y_pred,'r--')
    #----
    plt.yticks(fontsize=font_size)
    plt.xticks(fontsize=font_size)
    plt.ylim(0,410)
    plt.xlim(0,410)
    plt.xlabel('Actual Parking Duration (min)',fontsize=font_size)
    plt.ylabel('Expected Parking Duration (min)',fontsize=font_size)
    plt.tight_layout()
    plt.show()    
    
def plot_income_young(data_2014,data_2015,save_fig=0):
    density_2014 = []
    density_2015 = []
    labels_list = ('<2.5','2.5~5.0','5.0~7.5','7.5~10','>10')
    data_2014 = data_2014.loc[data_2014['AGE']==1]
    data_2015 = data_2015.loc[data_2015['AGE']==1]
    columns = pd.unique(data_2015['INCOME'].loc[data_2015['INCOME']!=-1])
    columns = sorted(columns)
    #print (columns)
    total_2014 = len(data_2014['INCOME'].loc[data_2014['INCOME']!=-1])
    total_2015 = len(data_2015['INCOME'].loc[data_2015['INCOME']!=-1])
    
    for key in columns:
        num = len(data_2014['INCOME'].loc[data_2014['INCOME']==key])
        density_2014.append(num/total_2014)        
        num = len(data_2015['INCOME'].loc[data_2015['INCOME']==key])
        density_2015.append(num/total_2015)
    N = len(columns)
    ind = np.arange(N)  # the x locations for the groups
    width = 0.3      # the width of the bars
    fig, ax = plt.subplots(figsize=(8, 6))
    rects1 = ax.bar(ind, density_2014, width, color=colors[0])
    rects2 = ax.bar(ind+width, density_2015, width, color=colors[1])
    print ('----------------')
    print('income', columns)       
    print ('2014 income', density_2014)
    print ('2015 income', density_2015)    
    plt.yticks(fontsize=16)
    #plt.ylim(0.35,0.46)
    #ax.set_yticklabels([str(i) + '%' for i in range(40, 61, 10)])    
    ax.set_ylabel('Density',fontsize=16)
    ax.set_xticks(ind+0.5*width)
    ax.set_xticklabels(labels_list,fontsize=16)
    ax.set_xlabel('Individual Monthly Income for 18'+r'$\leq$'+'Age'+r'$\leq$'+'30 ('+'RMB'+' 1000)',fontsize=16)
    ax.legend( (rects1[0], rects2[0]) , labels, fontsize=16, loc='upper right')
    plt.tight_layout()
    if save_fig == 0:
        plt.show()
    else:
        plt.savefig('img/Income_2014_2015_YOUNG.png', dpi=300)

def plot_satisfaction(save_fig=0):
    density_2014 = []
    density_2015 = []
    labels_list = ('Unsatisfied', 'Satisfied')
    columns = pd.unique(data_2015['EX_PLACE'].loc[data_2015['EX_PLACE'] != -1])
    columns = sorted(columns)
    # print (columns)
    total_2014 = len(data_2014['EX_PLACE'].loc[data_2014['EX_PLACE'] != -1])
    total_2015 = len(data_2015['EX_PLACE'].loc[data_2015['EX_PLACE'] != -1])

    for key in columns:
        num = len(data_2014['EX_PLACE'].loc[data_2014['EX_PLACE'] == key])
        density_2014.append(num / total_2014)
        num = len(data_2015['EX_PLACE'].loc[data_2015['EX_PLACE'] == key])
        density_2015.append(num / total_2015)
    N = len(columns)
    ind = np.arange(N)  # the x locations for the groups
    width = 0.3  # the width of the bars
    fig, ax = plt.subplots(figsize=(8, 6))
    rects1 = ax.bar(ind, density_2014, width, color=colors[0])
    rects2 = ax.bar(ind + width, density_2015, width, color=colors[1])
    print('----------------')
    print('EX_PLACE', columns)
    print('2014 EX_PLACE', density_2014)
    print('2015 EX_PLACE', density_2015)
    plt.yticks(fontsize=16)
    # plt.ylim(0.35,0.46)
    # ax.set_yticklabels([str(i) + '%' for i in range(40, 61, 10)])
    ax.set_ylabel('Density', fontsize=16)
    ax.set_xticks(ind + 0.5 * width)
    ax.set_xticklabels(labels_list, fontsize=16)
    # ax.set_xlabel('Gender', fontsize=16)
    ax.legend((rects1[0], rects2[0]), labels, fontsize=16)
    plt.tight_layout()
    if save_fig == 0:
        plt.show()
    else:
        plt.savefig('img/Satisfied_or_not_2014_2015.png', dpi=300)




def plot_three_sub_dimention(save_fig=0):

    labels_list_dict = {'Distance':['CHOICE',('Close', 'Moderate','Far')],'Price':['CHARGE',('Low', 'Moderate','High')],
                   'Vacancy':['SPARSE',('Crowding', 'Moderate', 'Vacant')]}

    for dep in labels_list_dict:
        density_2014 = []
        density_2015 = []
        var = labels_list_dict[dep][0]
        labels_list = labels_list_dict[dep][1]
        columns = pd.unique(data_2015[var].loc[data_2015[var] != -1])
        columns = sorted(columns)
        # print (columns)
        total_2014 = len(data_2014[var].loc[data_2014[var] != -1])
        total_2015 = len(data_2015[var].loc[data_2015[var] != -1])

        for key in columns:
            num = len(data_2014[var].loc[data_2014[var] == key])
            density_2014.append(num / total_2014)
            num = len(data_2015[var].loc[data_2015[var] == key])
            density_2015.append(num / total_2015)
        N = len(columns)
        ind = np.arange(N)  # the x locations for the groups
        width = 0.3  # the width of the bars
        fig, ax = plt.subplots(figsize=(6, 6))
        rects1 = ax.bar(ind, density_2014, width, color=colors[0])
        rects2 = ax.bar(ind + width, density_2015, width, color=colors[1])

        plt.yticks(fontsize=15)
        # plt.ylim(0.35,0.46)
        # ax.set_yticklabels([str(i) + '%' for i in range(40, 61, 10)])
        ax.set_ylabel('Density', fontsize=15)
        ax.set_xticks(ind + 0.5 * width)
        ax.set_xticklabels(labels_list, fontsize=15)
        ax.set_xlabel(dep, fontsize=16)
        ax.legend((rects1[0], rects2[0]), labels, fontsize=16)
        plt.tight_layout()
        if save_fig == 0:
            plt.show()
        else:
            plt.savefig('img/' + dep + '_2014_2015.png', dpi=300)

def calculate_statistics(data_2014, data_2015):
    # trip purpose:
    # working,
    # 2014
    def trip_purpose_sta(data_):
        work_num = len(data_.loc[data_['WORKING'] == 1])
        other = len(data_) - work_num
        return work_num, other

    work_num, other = trip_purpose_sta(data_2014)
    print('2014')
    print('work prop', work_num/(work_num  + other))
    print('other', other / (work_num  + other))
    work_num, other = trip_purpose_sta(data_2015)
    print('2015')
    print('work prop', work_num/(work_num  + other))
    print('other', other / (work_num  + other))

    a=1

if __name__ == '__main__':
    #------
    #plot_income(save_fig=1)
    #------
    #plot_age(save_fig=0)
    #------
    #plot_gender(save_fig=0)
    #------
    #proportions()
    #---------
    # area_number()
    #----
    #parking_related_variables(data_2014,if_2014=True)
    #print('--------**********------------2015----')
    #parking_related_variables(data_2015,if_2014=False)
    #----
    #parking_characteristic(save_fig=0)
    #------------
    #scatter_duration()
    #-----
    #plot_income_young(data_2014,data_2015,save_fig=1)
    #--------------

    # plot_satisfaction(save_fig=1)
    # plot_three_sub_dimention(save_fig=1)

    ###############
    calculate_statistics(data_2014, data_2015)