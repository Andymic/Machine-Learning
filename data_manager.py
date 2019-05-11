# @Author: floki
# @Date:   2019-05-11T14:25:54-04:00
# @Last modified by:   floki
# @Last modified time: 2019-05-11T14:48:43-04:00

import os
import tarfile
from six.moves import urllib
import pandas as pd

DEFAULT_URL = 'https://raw.githubusercontent.com/ageron/handson-ml/master'
DEFAULT_PATH = 'data'
def fetch_housing_data(path=DEFAULT_PATH):
    url = DEFAULT_URL + 'datasets/housing/housing.tgz
    path = path + '/housing'
    if not os.path.isdir(path):
        print('Housing data directory does not exist, downloading...')
        os.makedirs(path)
        tgz_path = os.path.join(path, 'housing.tgz')
        urllib.request.urlretrieve(url, tgz_path)
        housing_tgz = tarfile.open(tgz_path)
        housing_tgz.extractall(path=path)
        housing_tgz.close()

    csv_path = os.path.join(path, 'housing.csv')
    return pd.read_csv(csv_path)
