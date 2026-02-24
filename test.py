import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------------Function------------------
def normalize(x):
    res = (x - np.mean(x)) / np.std(x)
    return res

# --------------- Data ---------------------
rows = 100
data = pd.read_csv("used_cars.csv", nrows=rows)

# ---------- Data cleaning ---------------
model_year_data = data['model_year'].astype(int)
milage_data = data['milage'].str.replace(' mi.', '').str.replace(',','').astype(int)
accident_data = data['accident'].str.replace('At least 1 accident or damage reported', '1').str.replace('None reported', '0')
accident_data = accident_data.fillna(0).astype(int)
price_data = data['price'].str.replace('$', '').str.replace(',','').astype(int)

x_year = np.array(model_year_data)
x_milage = np.array(milage_data)
x_accident = np.array(accident_data)
y_price = np.array(price_data)

# ---------- data normalization ------------
x_year_z = normalize(x_year)
x_milage_z = normalize(x_milage)
x_accident_z = normalize(x_accident)
y_price_z = normalize(y_price)

# ------------ Regression -----------------
y_z = y_price_z.reshape(-1, 1)

ones = np.ones_like(x_year_z.reshape(-1,1))
mat_x = np.concatenate((ones, x_year_z.reshape(-1,1), x_year_z.reshape(-1,1)**2), axis=1)

first_term = np.linalg.inv(mat_x.T @ mat_x)
sec_term = mat_x.T @ y_z
result = first_term @ sec_term
result = result.flatten()

print(result)
x_year_z = np.sort(x_year_z)
plt.plot(x_year_z, y_z, 'bo')
plt.plot(x_year_z, result[0] + result[1]*x_year_z + result[2]*(x_year_z**2), '-r')
plt.show()

# -------------- Input ----------------
year = int(input("Enter the year: "))
year_normalized = float((year - np.mean(x_year)) / np.std(x_year))

x_input = np.array([1, year_normalized, year_normalized**2])
price_normalized = x_input @ result
price = (price_normalized*np.std(y_price)) + np.mean(y_price)
print(float(price))