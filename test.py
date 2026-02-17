import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --------------- Data ---------------------
rows = 5
data = pd.read_csv("used_cars.csv", nrows=rows)

model_year_data = data['model_year'].astype(int)
milage_data = data['milage'].str.replace(' mi.', '').str.replace(',','').astype(int)
accident_data = data['accident'].str.replace('At least 1 accident or damage reported', '1').str.replace('None reported', '0').astype(int)
price_data = data['price'].str.replace('$', '').str.replace(',','').astype(int)
