import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# 1. Load dataset
data = pd.read_csv('data_banjir.csv')

# 2. Pilih fitur yang mau dipakai
X = data[['Tavg', 'RH_avg', 'RR']]
y = data['flood']

# 3. Bersihkan data
data_bersih = pd.concat([X, y], axis=1).dropna()
X = data_bersih[['Tavg', 'RH_avg', 'RR']]
y = data_bersih['flood']

# 4. Bagi data train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Normalisasi
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Bangun model ANN
model = Sequential([
    Dense(64, activation='relu', input_shape=(3,)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 7. Training
model.fit(X_train_scaled, y_train, epochs=50, batch_size=8, validation_data=(X_test_scaled, y_test))

# 8. Save model dan scaler
model.save('model_banjir.h5')
joblib.dump(scaler, 'scaler.save')

print("Model dan scaler berhasil disimpan!")
