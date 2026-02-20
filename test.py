import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --------------- Data ---------------------
rows = 30
data = pd.read_csv("used_cars.csv", nrows=rows)

# ------ Data cleaning ---------------
model_year_data = data['model_year'].astype(int)
milage_data = data['milage'].str.replace(' mi.', '').str.replace(',','').astype(int)
accident_data = data['accident'].str.replace('At least 1 accident or damage reported', '1').str.replace('None reported', '0')
accident_data = accident_data.fillna(0).astype(int)
price_data = data['price'].str.replace('$', '').str.replace(',','').astype(int)

# plt.scatter(model_year_data, price_data)
# plt.show()

# ------------- Standardization ----------------
x = np.array(model_year_data)
y = np.array(price_data)

# x = np.array([1,2,3,4,5,6,7,8,9,10])
# y = np.array([2,4,6,8,10,12,14,16,18,20])

# Z score normalization
x = (x - np.mean(x)) / np.std(x)
y = (y - np.mean(y)) / np.std(y)

# Min Max normalization
# x = (x - np.min(x)) / (np.max(x) - np.min(x))
# y = (y - np.min(y)) / (np.max(y) - np.min(y))

# ------------ Regression -----------------
y = y.reshape(-1, 1)

ones = np.ones_like(x.reshape(-1,1))
mat_x = np.concatenate((x.reshape(-1,1), ones), axis=1)

first_term = np.linalg.inv(mat_x.T @ mat_x)
sec_term = mat_x.T @ y
result = first_term @ sec_term

print(result)
plt.plot(x,y, 'bo')
plt.plot(x, result[0]*x + result[1], '-r')
plt.show()