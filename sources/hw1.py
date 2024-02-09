import urllib.request as download
import pandas
import os

download_url = 'https://raw.githubusercontent.com/tdhock/2023-08-deep-learning/main/data/zip.test.gz'
download_path = 'd:/home/data.test.gz'

if not os.path.isfile(download_path):
    download.urlretrieve(download_url,download_path)
    print('File downloaded successfully')
else:
    print('File already exists')

df = pandas.read_csv(download_path,sep=' ',header=None)
print("dataframe shape: ",df.shape)
