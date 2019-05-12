# @Author: floki
# @Date:   2019-05-12T16:47:00-04:00
# @Last modified by:   floki
# @Last modified time: 2019-05-12T17:24:20-04:00

import data_manager as dm
import matplotlib.pyplot as plt

def housing_project():
    data = dm.load_housing_data()
    print(data.head())
    print(data.info())

    data.hist(bins=50, figsize=(20,15))
    plt.show()





if __name__ == '__main__':
    housing_project()
