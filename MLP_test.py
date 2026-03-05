import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# --------------- Variables ---------------------
rows = 180
test_n = 20
data = pd.read_csv("used_cars.csv", nrows=rows)

# ---------- Data cleaning ---------------
# Train set
model_year_data = data['model_year'].astype(int)
milage_data = data['milage'].str.replace(' mi.', '').str.replace(',', '').astype(int)
accident_data = data['accident'].str.replace('At least 1 accident or damage reported', '1').str.replace('None reported', '0')
accident_data = accident_data.fillna(0).astype(int)
price_data = data['price'].str.replace('$', '').str.replace(',', '').astype(int)

x_year = np.array(model_year_data)
x_milage = np.array(milage_data)
x_accident = np.array(accident_data)
y_price = np.array(price_data)

# Test set
rows_test = rows + test_n
test = pd.read_csv("used_cars.csv", nrows=rows_test)
test = test.tail(test_n)

model_year_test = test['model_year'].astype(int)
milage_test = test['milage'].str.replace(' mi.', '').str.replace(',', '').astype(int)
accident_test = test['accident'].str.replace('At least 1 accident or damage reported', '1').str.replace('None reported', '0')
accident_test = accident_test.fillna(0).astype(int)
price_test = test['price'].str.replace('$', '').str.replace(',', '').astype(int)

x_year_test = np.array(model_year_test)
x_milage_test = np.array(milage_test)
x_accident_test = np.array(accident_test)
y_price_test = np.array(price_test)

# ---------- Assemble feature matrices ------------
X_train = np.column_stack((x_year, x_milage, x_accident))
X_test  = np.column_stack((x_year_test, x_milage_test, x_accident_test))

# ---------- Normalization ------------
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_z = scaler_X.fit_transform(X_train)
X_test_z  = scaler_X.transform(X_test)

y_train_z = scaler_y.fit_transform(y_price.reshape(-1, 1)).flatten()
y_test_z  = scaler_y.transform(y_price_test.reshape(-1, 1)).flatten()

# -------------- MLP Regressor -----------------
mlp = MLPRegressor(
    hidden_layer_sizes=(64, 32),
    activation='relu',
    solver='adam',
    max_iter=2000,
    learning_rate_init=0.001,
    random_state=42
)

mlp.fit(X_train_z, y_train_z)
print(f"Training converged in {mlp.n_iter_} iterations")

# ----------- Accuracy Metrics ---------
def acc(y_pred_z):
    mse  = mean_squared_error(y_test_z, y_pred_z)
    rmse = np.sqrt(mse)
    r_sq = r2_score(y_test_z, y_pred_z)
    print(f"\n---- Accuracy Metrics for MLP ----")
    print(f"MSE:       {mse:.4f}")
    print(f"RMSE:      {rmse:.4f}")
    print(f"R_SQUARED: {r_sq:.4f}")

y_pred_z = mlp.predict(X_test_z)
acc(y_pred_z)

# -------------- Input ----------------
# print("\n--- Car price prediction ---")
# year     = int(input("Enter the year: "))
# mile     = int(input("Enter the milage: "))
# accident = int(input("Are you okay if the car has been in an accident (1 for okay, 0 for not okay): "))

# x_input_z = scaler_X.transform([[year, mile, accident]])
# price_normalized = mlp.predict(x_input_z)[0]
# price = scaler_y.inverse_transform([[price_normalized]])[0][0]
# print(f"The car would cost: ${price:,.2f}")