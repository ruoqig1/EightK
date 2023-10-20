import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from tensorflow import keras
from tensorflow.keras import regularizers, callbacks

# Create synthetic dataset
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Logistic Regression with scikit-learn using ElasticNet penalty
clf = LogisticRegression(penalty='elasticnet', l1_ratio=0.5, solver='saga', max_iter=10000)
clf.fit(X_train, y_train)
y_pred_sklearn = clf.predict(X_test)
accuracy_sklearn = accuracy_score(y_test, y_pred_sklearn)
print(f"Accuracy of scikit-learn Logistic Regression (with ElasticNet): {accuracy_sklearn:.4f}")

# Logistic Regression with TensorFlow using ElasticNet regularization
l1_l2_reg = regularizers.l1_l2(l1=0.01, l2=0.01)  # Modify l1 and l2 values as needed

model = keras.Sequential([
    keras.layers.Dense(1, input_dim=X_train.shape[1], activation='sigmoid', kernel_regularizer=l1_l2_reg)
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# Define early stopping callback
early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
# 'patience' defines the number of epochs to wait before stopping if no improvement is seen

# Split a portion of the training data for validation
validation_split = 0.1  # 10% of training data used for validation

model.fit(X_train, y_train, epochs=1000, batch_size=32, verbose=True, validation_split=validation_split, callbacks=[early_stopping])

_, accuracy_tensorflow = model.evaluate(X_test, y_test, verbose=0)
print(f"Accuracy of TensorFlow Logistic Regression (with ElasticNet): {accuracy_tensorflow:.4f}")
