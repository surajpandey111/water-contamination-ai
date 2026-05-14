import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ==============================
# LOAD DATASET
# ==============================

print("Loading Dataset...")

data = pd.read_csv("water_potability.csv")

print("\nFirst 5 Rows:")
print(data.head())

# ==============================
# HANDLE MISSING VALUES
# ==============================

print("\nHandling Missing Values...")

data.fillna(data.mean(), inplace=True)

# ==============================
# FEATURES AND TARGET
# ==============================

X = data.drop("Potability", axis=1)
y = data["Potability"]

# ==============================
# FEATURE SCALING
# ==============================

print("\nScaling Data...")

scaler = StandardScaler()

X = scaler.fit_transform(X)
joblib.dump(scaler, "scaler.pkl")
# ==============================
# TRAIN TEST SPLIT
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))

# ==============================
# BUILD DEEP LEARNING MODEL
# ==============================

print("\nBuilding Deep Learning Model...")

model = Sequential([

    Dense(128, activation='relu', input_shape=(X.shape[1],)),

    Dropout(0.3),

    Dense(64, activation='relu'),

    Dropout(0.3),

    Dense(32, activation='relu'),

    Dense(1, activation='sigmoid')
])

# ==============================
# COMPILE MODEL
# ==============================

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# ==============================
# EARLY STOPPING
# ==============================

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

# ==============================
# TRAIN MODEL
# ==============================

print("\nTraining Model...\n")

history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stop]
)

# ==============================
# SAVE MODEL
# ==============================

model.save("model.keras")

print("\nModel Saved Successfully!")

# ==============================
# MODEL EVALUATION
# ==============================

print("\nEvaluating Model...")

loss, accuracy = model.evaluate(X_test, y_test)

print(f"\nTest Accuracy: {accuracy * 100:.2f}%")
print(f"Test Loss: {loss:.4f}")

# ==============================
# PREDICTIONS
# ==============================

y_pred = model.predict(X_test)

# Convert probabilities to binary
y_pred = (y_pred > 0.5).astype(int)

# ==============================
# CONFUSION MATRIX
# ==============================

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# ==============================
# ACCURACY GRAPH
# ==============================

plt.figure(figsize=(8, 5))

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.legend([
    'Training Accuracy',
    'Validation Accuracy'
])

plt.grid(True)

plt.savefig("accuracy_graph.png")

plt.show()

# ==============================
# LOSS GRAPH
# ==============================

plt.figure(figsize=(8, 5))

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.legend([
    'Training Loss',
    'Validation Loss'
])

plt.grid(True)

plt.savefig("loss_graph.png")

plt.show()

# ==============================
# PIE CHART
# ==============================

safe_count = len(data[data['Potability'] == 1])
unsafe_count = len(data[data['Potability'] == 0])

labels = ['Safe Water', 'Contaminated Water']
sizes = [safe_count, unsafe_count]

plt.figure(figsize=(6, 6))

plt.pie(
    sizes,
    labels=labels,
    autopct='%1.1f%%'
)

plt.title("Water Quality Distribution")

plt.savefig("water_distribution.png")

plt.show()

# ==============================
# FINAL MESSAGE
# ==============================

print("\n====================================")
print("PROJECT TRAINING COMPLETED")
print("====================================")

print("\nGenerated Files:")
print("1. model.keras")
print("2. accuracy_graph.png")
print("3. loss_graph.png")
print("4. water_distribution.png")

print("\nDeep Learning Water Contamination Detection System Ready!")