# @Author: Michel Andy <ragnar>
# @Date:   2020-01-25T10:32:50-05:00
# @Email:  Andymic12@gmail.com
# @Filename: life_satisfaction.py
# @Last modified by:   ragnar
# @Last modified time: 2020-01-25T11:42:32-05:00

import numpy as np
import os
from matplotlib import pyplot as plt
import pandas as pd
import sklearn.linear_model
import sklearn.neighbors
from utils.data_funcs import *
import config

data_dir = os.path.join(config.DATA_DIR, 'lifesat', '')

oecd_bli = pd.read_csv(data_dir + "oecd_bli_2015.csv", thousands=',')
gdp_per_capita = pd.read_csv(data_dir + "gdp_per_capita.csv",thousands=',',delimiter='\t',
                             encoding='latin1', na_values="n/a")

country_stats = prepare_country_stats(oecd_bli, gdp_per_capita)
X = np.c_[country_stats["GDP per capita"]]
y = np.c_[country_stats["Life satisfaction"]]

country_stats.plot(kind='scatter', x="GDP per capita", y='Life satisfaction')

type = 'is'
model = None

if type == 'mb':
    print('Model-Based Learning')
    model = sklearn.linear_model.LinearRegression()
else:
    print('Instance-Based Learning')
    model = sklearn.neighbors.KNeighborsRegressor(n_neighbors=4)
model.fit(X, y)

X_new = [[60000]]
print(model.predict(X_new))

plt.show()
