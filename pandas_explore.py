import pandas as pd
import numpy as np


train = pd.read_csv("data/ames-house-prices/train.csv")


print("train : " + str(train.shape))


train.SalePrice = np.log1p(train.SalePrice)


small = train[["SalePrice", "GrLivArea"]]

corr = small.corr()
corr.sort_values(["SalePrice"], ascending = False, inplace = True)
print(corr.SalePrice)
