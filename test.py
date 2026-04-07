import urllib.request

url = "https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv"
print(urllib.request.urlopen(url).status)
