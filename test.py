import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

rows = 5

a = pd.read_csv("used_cars.csv", nrows=rows)
print(a)

brand = pd.read_csv("used_cars.csv", nrows=rows, usecols=[0])
# print(brand)
model = pd.read_csv("used_cars.csv", nrows=rows, usecols=[1])
model_year = pd.read_csv("used_cars.csv", nrows=rows, usecols=[2])
mile = pd.read_csv("used_cars.csv", nrows=rows, usecols=[3])

price = pd.read_csv("used_cars.csv", nrows=rows, usecols=[11])
print(price)