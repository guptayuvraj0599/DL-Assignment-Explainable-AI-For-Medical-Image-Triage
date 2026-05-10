import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# from tensorflow import keras

# from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)

import seaborn as sns

# ==============================
# DATASET PATHS
# ==============================

TRAIN_DIR = "dataset/train"
VAL_DIR = "dataset/val"
TEST_DIR = "dataset/test"

# ==============================
# IMAGE GENERATORS
# ==============================

train_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    zoom_range=0.1,
    horizontal_flip=True
)

val_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
test_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    TRAIN_DIR,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    shuffle=True
)

val_data = val_gen.flow_from_directory(
    VAL_DIR,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

test_data = test_gen.flow_from_directory(
    TEST_DIR,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

# ==============================
# MODEL
# ==============================

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224,224,3),
    include_top=False,
    weights='imagenet'
)

base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation='sigmoid')
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
# CALLBACKS
# ==============================

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "best_model.h5",
    monitor='val_accuracy',
    save_best_only=True,
    mode='max'
)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

# ==============================
# TRAIN MODEL
# ==============================

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=15,
    callbacks=[checkpoint, early_stop]
)

# ==============================
# SAVE FINAL MODEL
# ==============================

model.save("model.h5")

# ==============================
# EVALUATE MODEL
# ==============================

test_loss, test_acc = model.evaluate(test_data)

print(f"\nTest Accuracy: {test_acc*100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")

# ==============================
# PREDICTIONS
# ==============================

pred_probs = model.predict(test_data)
predictions = (pred_probs > 0.5).astype(int)

true_labels = test_data.classes

# ==============================
# CLASSIFICATION REPORT
# ==============================

print("\nClassification Report:\n")

report = classification_report(
    true_labels,
    predictions,
    target_names=['NORMAL', 'PNEUMONIA']
)

print(report)

# ==============================
# CONFUSION MATRIX
# ==============================

cm = confusion_matrix(true_labels, predictions)

plt.figure(figsize=(6,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['NORMAL', 'PNEUMONIA'],
            yticklabels=['NORMAL', 'PNEUMONIA'])

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.savefig("confusion_matrix.png")
plt.close()

# ==============================
# ROC CURVE
# ==============================

fpr, tpr, thresholds = roc_curve(true_labels, pred_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0,1], [0,1], linestyle='--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()

plt.savefig("roc_curve.png")
plt.close()

print(f"\nAUC-ROC Score: {roc_auc:.4f}")

# ==============================
# ACCURACY CURVE
# ==============================

plt.figure(figsize=(8,6))

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy Curve")
plt.legend()

plt.savefig("accuracy_curve.png")
plt.close()

# ==============================
# LOSS CURVE
# ==============================

plt.figure(figsize=(8,6))

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.legend()

plt.savefig("loss_curve.png")
plt.close()

print("\nTraining Complete.")
print("All graphs saved successfully.")