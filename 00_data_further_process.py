'''
Plot before and after data analysis curve
'''
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
#import seaborn as sns

#----pre_process
#data_2014 = pd.read_excel('data/data_2014_format.xlsx')
#data_2015 = pd.read_excel('data/data_2015_format.xlsx')
#data_201415 = pd.read_excel('data/data_combined1415_format.xlsx')
#data_2014.to_csv('data/data_2014.csv',index=False)
#data_2015.to_csv('data/data_2015.csv',index=False)
#data_201415.to_csv('data/data_201415.csv',index=False)
#-----------------------
data_2014 = pd.read_csv('data/data_2014.csv')
data_2015 = pd.read_csv('data/data_2015.csv')
#data_201415 = pd.read_csv('data/data_201415.csv')

def process_data(data_2014):
    print (len(data_2014))
    data_2014 = data_2014.loc[(data_2014['CHARGE']!=-1)]
    print (len(data_2014))
    data_2014 = data_2014.loc[(data_2014['SELF_PAY']!=-1)]
    print (len(data_2014))
    data_2014 = data_2014.loc[(data_2014['HAVE_PARKING']!=-1)]
    print (len(data_2014))
    data_2014 = data_2014.loc[(data_2014['SELF_CAR']!=-1)]
    print (len(data_2014))
    data_2014 = data_2014.loc[(data_2014['ILL_NEVER']!=-1)]
    print (len(data_2014))
    data_2014 = data_2014.loc[(data_2014['WORKING']!=-1)]
    print (len(data_2014))
    data_2014 = data_2014.loc[(data_2014['INCOME']!=-1)]
    print (len(data_2014))
    data_2014 = data_2014.loc[(data_2014['AGE']!=-1)]
    print (len(data_2014))
    data_2014 = data_2014.loc[(data_2014['MALE']!=-1)]
    print (len(data_2014))
    data_2014 = data_2014.loc[(data_2014['AP_FEE']!=-1)]
    print (len(data_2014))
    data_2014 = data_2014.loc[(data_2014['TEMP']!=-1)]
    print (len(data_2014))    
    data_2014.loc[(data_2014['SUN']==-1),'SUN'] = 0
    #print (len(data_2014))    
    return data_2014

def add_unit_price():
    data_2014_new = data_2014_new.loc[(data_2014_new['AREA']==1)| 
                                      (data_2014_new['AREA']==2)].copy() # only reserve data from area A and B
    data_2015_new = data_2015_new.loc[(data_2015_new['AREA']==1)| 
                                      (data_2015_new['AREA']==2)].copy() # only reserve data from area A and B    print (len(data_new))
    data_2014_new['UNIT_PRICE'] = 0
    data_2015_new['UNIT_PRICE'] = 0
    
    data_2014_new['UNIT_PRICE'] = 0
    data_2014_new.loc[data_2014_new['AREA']==1,'UNIT_PRICE'] = 3 #2014 A
    data_2014_new.loc[data_2014_new['AREA']==2,'UNIT_PRICE'] = 2.5 #2014 B

    data_2015_new.loc[data_2015_new['AREA']==1,'UNIT_PRICE'] = 5 #2015 A
    data_2015_new.loc[data_2015_new['AREA']==2,'UNIT_PRICE'] = 4 #2015 B    
    
def combine(data_2014_new, data_2015_new):
    data_2014_new['S2014'] = 1
    data_2014_new['S2015'] = 0
    data_2015_new['S2015'] = 1
    data_2015_new['S2014'] = 0
    columns_2014 = set(list(data_2014_new.columns))
    columns_2015 = set(list(data_2015_new.columns))
    columns1415 = columns_2014.intersection(columns_2015)
    data_2014_new = data_2014_new.loc[:,list(columns1415)]
    data_2015_new = data_2015_new.loc[:,list(columns1415)]
    data_1415_new = pd.concat([data_2014_new, data_2015_new])
    return data_1415_new
if __name__ == '__main__':
    #data_2014_new = process_data(data_2014)
    #data_2015_new = process_data(data_2015)
    ##--------------

    
    #data_2014_new.to_csv('data/data_2014_new.csv',index = False)
    #data_2015_new.to_csv('data/data_2015_new.csv',index = False)
    #========Combine------------
    data_2014_new = pd.read_csv('data/data_2014_new.csv')
    data_2015_new = pd.read_csv('data/data_2015_new.csv') 
    data_1415_new = combine(data_2014_new, data_2015_new)
    data_1415_new.to_csv('data/data_1415_new.csv',index = False)