import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

rows = 100
data = pd.read_csv("used_cars.csv", nrows=rows)

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

raw_dataset = np.concatenate((y_price.reshape(-1,1), x_year.reshape(-1,1), x_milage.reshape(-1,1), x_accident.reshape(-1,1)), axis = 1)
raw_dataset = pd.DataFrame(raw_dataset)
print(raw_dataset.head())

dataset = raw_dataset.copy()
dataset.tail()

train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

sns.pairplot(train_dataset['price', 'year', 'milage', 'accident'], diag_kind='kde')
train_dataset.describe().transpose()

train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop('price')
test_labels = test_features.pop('price')

train_dataset.describe().transpose()[['mean', 'std']]

normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(np.array(train_features))
print(normalizer.mean.numpy())

first = np.array(train_features[:1])

with np.printoptions(precision=2, suppress=True):
    print('First example:', first)
    print()
    print('Normalized:', normalizer(first).numpy())

year = np.array(train_features['year'])

year_normalizer = layers.Normalization(input_shape=[1,], axis=None)
year_normalizer.adapt(year)

year_model = tf.keras.Sequential([
    year_normalizer,
    layers.Dense(units=1)
])

year_model.summary()
year_model.predict(year[:10])

year_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
    loss='mean_absolute_error')

history = year_model.fit(
    train_features['year'],
    train_labels,
    epochs=100,
    # Suppress logging.
    verbose=0,
    # Calculate validation results on 20% of the training data.
    validation_split = 0.2)

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()

def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim([0, 10])
    plt.xlabel('Epoch')
    plt.ylabel('Error [price]')
    plt.legend()
    plt.grid(True)

plot_loss(history)

test_results = {}

test_results['year_model'] = year_model.evaluate(
    test_features['year'],
    test_labels, verbose=0)

x = tf.linspace(0.0, 250, 251)
y = year_model.predict(x)

def plot_year(x, y):
    plt.scatter(train_features['year'], train_labels, label='Data')
    plt.plot(x, y, color='k', label='Predictions')
    plt.xlabel('year')
    plt.ylabel('price')
    plt.legend()