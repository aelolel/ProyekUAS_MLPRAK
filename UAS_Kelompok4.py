import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_excel('BlaBla.xlsx', sheet_name='Sheet1')

# Define variables - ensure we're only selecting numeric columns
numeric_cols = ['UMUR_TAHUN'] + [f'{chr(i)}' for i in range(66, 77) if f'{chr(i)}' in df.columns]
X = df[numeric_cols]
y = df['N']  # Output variable

# a. Check Missing Values and Outliers
print("Missing values:\n", X.isnull().sum())
print("\nOutlier detection (summary statistics):\n", X.describe())

# Handle missing values - only for numeric columns
for col in X.select_dtypes(include=['int64', 'float64']).columns:
    if X[col].isnull().sum() > 0:
        X[col].fillna(X[col].median(), inplace=True)

# Convert any non-numeric columns to numeric (if they contain 'Y'/'N' etc.)
for col in X.columns:
    if X[col].dtype == object:
        # Convert 'Y' to 1 and others to 0, or use more sophisticated encoding
        X[col] = X[col].apply(lambda x: 1 if str(x).upper() == 'Y' else 0)

# Visualize age distribution before encoding
plt.figure(figsize=(10,5))
plt.hist(X['UMUR_TAHUN'], bins=20)
plt.title('Age Distribution Before Encoding')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# b. Age Encoding
def encode_age(age):
    if age <= 20: return 1
    elif 21 <= age <= 30: return 2
    elif 31 <= age <= 40: return 3
    elif 41 <= age <= 50: return 4
    else: return 5

X['UMUR_ENCODED'] = X['UMUR_TAHUN'].apply(encode_age)
X = X.drop('UMUR_TAHUN', axis=1)  # Remove original age column

# Check class distribution
print("\nClass distribution (N):\n", y.value_counts())

# Select top features using chi-square
selector = SelectKBest(chi2, k='all')  # First get all scores
selector.fit(X, y)

# Get scores and p-values
feature_scores = pd.DataFrame({
    'Feature': X.columns,
    'Score': selector.scores_,
    'p-value': selector.pvalues_
}).sort_values('Score', ascending=False)

print("\nFeature scores from Chi-square test:")
print(feature_scores)

# Select top 8 features based on scores
top_features = feature_scores.head(8)['Feature'].values
X = X[top_features]

# 4. Imbalanced Data - Under Sampling
print("\nBefore undersampling class counts:", y.value_counts())
undersampler = RandomUnderSampler(random_state=42)
X_under, y_under = undersampler.fit_resample(X, y)
print("After undersampling class counts:", pd.Series(y_under).value_counts())

# 5. Classification - Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X_under, y_under, test_size=0.2, random_state=42, stratify=y_under)

# Deep Learning Model
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy', 
                       keras.metrics.Precision(name='precision'),
                       keras.metrics.Recall(name='recall')])

# Add early stopping
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(X_train, y_train,
                    epochs=100,
                    batch_size=32,
                    validation_split=0.2,
                    callbacks=[early_stopping],
                    verbose=1)

# Plot training history
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

# 6. Evaluation
y_pred = (model.predict(X_test) > 0.5).astype("int32")

print("\nScenario 1 (Under Sampling) Results:")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("AUC-ROC:", roc_auc_score(y_test, y_pred))

# 7. Imbalanced Data - Over Sampling
print("\nBefore oversampling class counts:", y.value_counts())
oversampler = RandomOverSampler(random_state=42)
X_over, y_over = oversampler.fit_resample(X, y)
print("After oversampling class counts:", pd.Series(y_over).value_counts())

# 8. Classification - Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X_over, y_over, test_size=0.2, random_state=42, stratify=y_over)

# Deep Learning Model (with same architecture for fair comparison)
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy',
                       keras.metrics.Precision(name='precision'),
                       keras.metrics.Recall(name='recall')])

history = model.fit(X_train, y_train,
                    epochs=100,
                    batch_size=32,
                    validation_split=0.2,
                    callbacks=[early_stopping],
                    verbose=1)

# Plot training history
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

# 9. Evaluation
y_pred = (model.predict(X_test) > 0.5).astype("int32")

print("\nScenario 2 (Over Sampling) Results:")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("AUC-ROC:", roc_auc_score(y_test, y_pred))