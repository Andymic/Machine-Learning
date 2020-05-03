# @Author: Michel Andy <ragnar>
# @Date:   2020-02-02T00:27:58-05:00
# @Email:  Andymic12@gmail.com
# @Filename: housing_prices.py
# @Last modified by:   ragnar
# @Last modified time: 2020-02-02T01:20:01-05:00
import os
import config
import tarfile
import pandas as pd
import matplotlib.pyplot as plt

def unpack_data():
    data_dir = os.path.join(config.DATA_DIR, 'housing', '')
    csv_path = os.path.join(config.DATA_DIR, 'housing', 'housing.csv')
    if not os.path.isfile(csv_path):
        tgz = tarfile.open(data_dir + '/housing.tgz')
        tgz.extractll(path=data_dir)
        tgz.close()

    return pd.read_csv(csv_path)

DATA = unpack_data()

def line():
    print('\n\n')

#shows labels and the first few records
print(DATA.head())
line()

#pd frame
print(DATA.info())
line()

#ocean_proximity is a categorical value (object)
print(DATA['ocean_proximity'].value_counts())
line()

#summary of the numerical attributes of the data
print('Data summary:')
print(DATA.describe())
line()

#shows the number of instances (on the vertical axis) that have
# a given value range (on the horizontal axis)
DATA.hist(bins=50, figsize=(20,15))
plt.show()
