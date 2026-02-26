import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --------------- Variables ---------------------
rows = 100
data = pd.read_csv("used_cars.csv", nrows=rows)

# ------------- Functions ------------------
def normalize(x):
    res = (x - np.mean(x)) / np.std(x)
    return res

def acc(y):
    e = y_price_test_z - y
    sse = e.T @ e
    mse = sse / rows
    print(f"MSE: {mse}")

    rmse = np.sqrt(mse)
    print(f"RMSE: {rmse}")

    num = y - y_price_test_z
    num = num.T @ num
    denom = y_price_test_z - np.mean(y_price_z)
    denom = denom.T @ denom
    r_sq = 1 - (num/denom)
    print(f"R_SQUARED: {r_sq}")

# ---------- Data cleaning ---------------
# Train set
model_year_data = data['model_year'].astype(int)
milage_data = data['milage'].str.replace(' mi.', '').str.replace(',','').astype(int)
accident_data = data['accident'].str.replace('At least 1 accident or damage reported', '1').str.replace('None reported', '0')
accident_data = accident_data.fillna(0).astype(int)
price_data = data['price'].str.replace('$', '').str.replace(',','').astype(int)

x_year = np.array(model_year_data)
x_milage = np.array(milage_data)
x_accident = np.array(accident_data)
y_price = np.array(price_data)

# Test set
test_n = 10
rows_test = rows + test_n
test = pd.read_csv("used_cars.csv", nrows=rows_test)
test = test.tail(test_n)

model_year_test = test['model_year'].astype(int)
milage_test = test['milage'].str.replace(' mi.', '').str.replace(',','').astype(int)
accident_test = test['accident'].str.replace('At least 1 accident or damage reported', '1').str.replace('None reported', '0')
accident_test = accident_test.fillna(0).astype(int)
price_test = test['price'].str.replace('$', '').str.replace(',','').astype(int)

x_year_test = np.array(model_year_test)
x_milage_test = np.array(milage_test)
x_accident_test = np.array(accident_test)
y_price_test = np.array(price_test)

# ---------- data normalization ------------
# Train set
x_year_z = normalize(x_year)
x_milage_z = normalize(x_milage)
x_accident_z = normalize(x_accident)
y_price_z = normalize(y_price)

# Test set
x_year_test_z = (x_year_test - np.mean(x_year)) / np.std(x_year)
x_milage_test_z = (x_milage_test - np.mean(x_milage)) / np.std(x_milage)
x_accident_test_z = (x_accident_test - np.mean(x_accident)) / np.std(x_accident)
y_price_test_z = (y_price_test - np.mean(y_price)) / np.std(y_price)

# ------------ Regression -----------------
y_z = y_price_z.reshape(-1, 1)

ones = np.ones_like(x_year_z.reshape(-1,1))
mat_x = np.concatenate((ones, x_year_z.reshape(-1,1), x_milage_z.reshape(-1,1), x_accident_z.reshape(-1,1)), axis=1)

first_term = np.linalg.inv(mat_x.T @ mat_x)
sec_term = mat_x.T @ y_z
result = first_term @ sec_term
result = result.flatten()

print(f"coefficient of normal: {result}")
# x_year_z = np.sort(x_year_z)
# plt.plot(x_year_z, y_z, 'bo')
# plt.plot(x_year_z, result[0] + result[1]*(x_year_z), '-r')
# plt.show()

# -------------- Weighted Regression -------------
# Covariance Matrix
p = 3
residual = y_z - (mat_x @ result.reshape(-1,1))
residual = residual.flatten()
weights = 1 / (residual**2)
weight_matrix = np.diag(weights)

first_term_2 = np.linalg.inv(mat_x.T @ weight_matrix @ mat_x)
sec_term_2 = mat_x.T @ weight_matrix @ y_z
coef = first_term_2 @ sec_term_2
coef = coef.flatten()

print(f"coefficient of weighted: {coef}")
# plt.plot(x_year_z, y_z, 'bo')
# plt.plot(x_year_z, coef[0] + coef[1]*(x_year_z), '-r')
# plt.show()

# ----------- Accuracy Matrics ---------
ones_test = np.ones(test_n)
mat_x_test = np.column_stack((ones_test, x_year_test_z, x_milage_test_z, x_accident_test_z))

print("\n---- Accuracy Matrics for normal ----")
y_pred = mat_x_test @ result
acc(y_pred)

print("\n---- Accuracy Matrics for weighted ----")
y_pred_weighted = mat_x_test @ coef
acc(y_pred_weighted)

# -------------- Input ----------------
print("\n--- Car price prediction ---")
year = int(input("Enter the year: "))
year_normalized = float((year - np.mean(x_year)) / np.std(x_year))
mile = int(input("Enter the milage: "))
mile_normalized = float((mile - np.mean(x_milage)) / np.std(x_milage))
accident = int(input("Are you okay if the car has been in an accident (1 for okay, 0 for not okay): "))
accident_normalized = float((accident - np.mean(x_accident)) / np.std(x_accident))

x_input = np.array([1, year_normalized, mile_normalized, accident_normalized])
price_normalized = x_input @ coef
price = (price_normalized*np.std(y_price)) + np.mean(y_price)
print(f"The car would costs: ${price}")