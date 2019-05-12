# @Author: floki
# @Date:   2019-05-12T16:47:00-04:00
# @Last modified by:   floki
# @Last modified time: 2019-05-12T17:02:08-04:00



import data_manager as dm

if __name__ == '__main__':
    data = dm.load_housing_data()
    print(data.head())
    print(data.info())
    
