import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --------------- Variables ---------------------
rows = 100
test_n = 10
data = pd.read_csv("used_cars.csv", nrows=rows)

# ------------- Functions ------------------
def normalize(x, x_cal):
    res = (x - np.mean(x_cal)) / np.std(x_cal)
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
    denom = y_price_test_z - np.mean(y_price_test_z)
    denom = denom.T @ denom
    r_sq = 1 - (num/denom)
    print(f"R_SQUARED: {r_sq}")

# ---------- Data cleaning ---------------
# Train set
model_year_data = data['model_year'].astype(int)
milage_data = data['milage'].str.replace(' mi.', '').str.replace(',','').astype(int)
price_data = data['price'].str.replace('$', '').str.replace(',','').astype(int)

x_year = np.array(model_year_data)
x_milage = np.array(milage_data)
y_price = np.array(price_data)

# Test set
rows_test = rows + test_n
test = pd.read_csv("used_cars.csv", nrows=rows_test)
test = test.tail(test_n)

model_year_test = test['model_year'].astype(int)
milage_test = test['milage'].str.replace(' mi.', '').str.replace(',','').astype(int)
price_test = test['price'].str.replace('$', '').str.replace(',','').astype(int)

x_year_test = np.array(model_year_test)
x_milage_test = np.array(milage_test)
y_price_test = np.array(price_test)

# ---------- data normalization ------------
# Train set
x_year_z = normalize(x_year, x_year)
x_milage_z = normalize(x_milage, x_milage)
y_price_z = normalize(y_price, y_price)

# Test set
x_year_test_z = normalize(x_year_test, x_year)
x_milage_test_z = normalize(x_milage_test, x_milage)
y_price_test_z = normalize(y_price_test, y_price)

# ------------ Regression -----------------
y_z = y_price_z.reshape(-1, 1)

ones = np.ones_like(x_year_z.reshape(-1,1))
mat_x = np.concatenate((ones, x_year_z.reshape(-1,1), x_milage_z.reshape(-1,1)), axis=1)

first_term = np.linalg.inv(mat_x.T @ mat_x)
sec_term = mat_x.T @ y_z
result = first_term @ sec_term
result = result.flatten()

print("---- Coefficient ----")
print(f"coefficient of normal: {result}")

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

# ----------- Accuracy Matrics ---------
ones_test = np.ones(test_n)
mat_x_test = np.column_stack((ones_test, x_year_test_z, x_milage_test_z))

print("\n---- Accuracy Matrics for normal ----")
y_pred = mat_x_test @ result
acc(y_pred)

print("\n---- Accuracy Matrics for weighted ----")
y_pred_weighted = mat_x_test @ coef
acc(y_pred_weighted)

# --------- Plot Graphs -----------
# Plane
year_grid, milage_grid = np.meshgrid(
    np.linspace(x_year_z.min(),   x_year_z.max(),   40),
    np.linspace(x_milage_z.min(), x_milage_z.max(), 40)
)

plane_ols = result[0] + result[1]*year_grid + result[2]*milage_grid
plane_wls = coef[0]   + coef[1]*year_grid   + coef[2]*milage_grid

# 3D PLOTS
fig = plt.figure(figsize=(15, 6))
fig.suptitle("Used Car Price Prediction", fontsize=11, fontweight='bold')

for idx, (plane, b, label, surf_color) in enumerate([
    (plane_ols, result, "Normal regression", "red"),
    (plane_wls, coef,   "Weighted regression", "red"),
], start=1):

    ax = fig.add_subplot(1, 2, idx, projection='3d')

    # --- Regression surface ---
    ax.plot_surface(year_grid, milage_grid, plane,
                    alpha=0.35, color=surf_color, edgecolor='none')

    # --- Train scatter ---
    ax.scatter(x_year_z, x_milage_z, y_price_z,
               c='blue', s=16, alpha=0.7, label='Train data', zorder=5)

    # --- Test scatter ---
    ax.scatter(x_year_test_z, x_milage_test_z, y_price_test_z,
               c='yellow', s=16, alpha=0.7, label='Test data', zorder=5)

    # --- Residual lines for test points ---
    mat_x_test = np.column_stack([np.ones(test_n), x_year_test_z, x_milage_test_z])
    y_pred_test = mat_x_test @ b
    for xyr, xmi, yact, ypred in zip(x_year_test_z, x_milage_test_z,
                                      y_price_test_z, y_pred_test):
        ax.plot([xyr, xyr], [xmi, xmi], [yact, ypred],
                color='gray', linewidth=0.8, alpha=0.6)

    ax.set_xlabel("Model Year (z)", labelpad=8)
    ax.set_ylabel("Mileage (z)",    labelpad=8)
    ax.set_zlabel("Price (z)",      labelpad=8)
    ax.set_title(label, fontsize=12, pad=10)
    ax.legend(loc='upper left', fontsize=8)
    ax.view_init(elev=22, azim=-50)

plt.tight_layout()
plt.show()