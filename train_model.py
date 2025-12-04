
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import matplotlib.pyplot as plt


CSV_PATH = "phishing_dataset.csv"  
TARGET_COL = "CLASS_LABEL"
DROP_COLS = ["id"]                
RANDOM_STATE = 42
EPOCHS = 40
BATCH_SIZE = 32



data = pd.read_csv(CSV_PATH)


for c in DROP_COLS:
    if c in data.columns:
        data = data.drop(columns=[c])


X = data.drop(columns=[TARGET_COL])
y = data[TARGET_COL].values

print("Feature shape:", X.shape)
print("Target distribution:\n", pd.Series(y).value_counts())


X_train, X_test, y_train, y_test = train_test_split(
    X.values, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


joblib.dump(scaler, "scaler.save")
print("Scaler saved as scaler.save")


input_dim = X_train.shape[1]

#Deep Neural Network (DNN)
model = Sequential([
    Dense(128, input_dim=input_dim, activation="relu"),
    Dropout(0.2),
    Dense(64, activation="relu"),
    Dropout(0.15),
    Dense(32, activation="relu"),
    Dense(1, activation="sigmoid")  # binary classification
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

# Train the model
history = model.fit(
    X_train, y_train,
    validation_split=0.15,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1 
)

# Evaluate on test data
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Loss: {loss:.4f}, Test Accuracy: {acc:.4f}")

# Save the trained model
model.save("phishing_model.h5")



plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.savefig("training_accuracy.png")
plt.show()
