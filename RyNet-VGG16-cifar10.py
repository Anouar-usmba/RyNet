# -*- coding: utf-8 -*-
import os, json, time, random
import numpy as np
import psutil
import tensorflow as tf

from tensorflow.keras import layers, models, Input, regularizers
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import os, json
import numpy as np
# ===========================
#  RyNet Fusion Module
# ===========================
class WeightedFeatureFusion(tf.keras.layers.Layer):
    def __init__(self, target_dim=256, **kwargs):
        name = kwargs.pop("name", "WFF")
        super(WeightedFeatureFusion, self).__init__(name=name, **kwargs)
        self.target_dim = target_dim
        # Projection layers (linear)
        self.proj_f1 = layers.Dense(target_dim, activation=None, kernel_regularizer=regularizers.l2(1e-4), name="proj_f1")
        self.proj_f2 = layers.Dense(target_dim, activation=None, kernel_regularizer=regularizers.l2(1e-4), name="proj_f2")
        self.proj_f3 = layers.Dense(target_dim, activation=None, kernel_regularizer=regularizers.l2(1e-4), name="proj_f3")
        # Attention layers (sigmoid)
        self.attn_f1 = layers.Dense(target_dim, activation='sigmoid', kernel_regularizer=regularizers.l2(1e-4), name="attn_f1")
        self.attn_f2 = layers.Dense(target_dim, activation='sigmoid', kernel_regularizer=regularizers.l2(1e-4), name="attn_f2")
        self.attn_f3 = layers.Dense(target_dim, activation='sigmoid', kernel_regularizer=regularizers.l2(1e-4), name="attn_f3")

    def call(self, inputs):
        f1_gap, f2_gap, f3_gap = inputs
        f1_proj = self.proj_f1(f1_gap); f2_proj = self.proj_f2(f2_gap); f3_proj = self.proj_f3(f3_gap)
        f1_weight = self.attn_f1(f1_gap); f2_weight = self.attn_f2(f2_gap); f3_weight = self.attn_f3(f3_gap)
        f1_transformed = f1_proj * f1_weight
        f2_transformed = f2_proj * f2_weight
        f3_transformed = f3_proj * f3_weight
        return f1_transformed + f2_transformed + f3_transformed

# ===========================
#  VGG16 (CIFAR-10) + RyNet



# Input shape (32x32 for CIFAR-10)
inputs = layers.Input(shape=(32, 32, 3))
l2 = regularizers.l2(1e-4)

# Block 1
x = layers.Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=l2)(inputs)
x = layers.BatchNormalization()(x)
f1 = layers.Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=l2)(x)
f1 = layers.BatchNormalization()(f1)
x = layers.MaxPooling2D(pool_size=(2, 2))(f1)

# Block 2
x = layers.Conv2D(128, (3, 3), padding='same', activation='relu', kernel_regularizer=l2)(x)
x = layers.BatchNormalization()(x)
f2 = layers.Conv2D(128, (3, 3), padding='same', activation='relu', kernel_regularizer=l2)(x)
f2 = layers.BatchNormalization()(f2)
x = layers.MaxPooling2D(pool_size=(2, 2))(f2)

# Block 3
x = layers.Conv2D(256, (3, 3), padding='same', activation='relu', kernel_regularizer=l2)(x)
x = layers.BatchNormalization()(x)
f3 = layers.Conv2D(256, (3, 3), padding='same', activation='relu', kernel_regularizer=l2)(x)
f3 = layers.BatchNormalization()(f3)


    # --- expose named tensors for interpretability ---
f1 = layers.Activation('linear', name='f1_feat')(f1)
f2 = layers.Activation('linear', name='f2_feat')(f2)
f3 = layers.Activation('linear', name='f3_feat')(f3)

# GAP on f1, f2, f3
f1_gap = layers.GlobalAveragePooling2D()(f1)
f2_gap = layers.GlobalAveragePooling2D()(f2)
f3_gap = layers.GlobalAveragePooling2D()(f3)

# Weighted Feature Fusion
fused_features = WeightedFeatureFusion()([f1_gap, f2_gap, f3_gap])
fused_features = layers.Activation('linear', name='fused_features')(fused_features)


# Final Classification Head (simple)
# x = layers.Dense(512, activation='relu', kernel_regularizer=l2)(weighted_features)
# x = layers.Dropout(0.5)(x)
x = layers.Dense(256, activation='relu', kernel_regularizer=l2)(fused_features)
x = layers.Dropout(0.5)(x)

outputs = layers.Dense(10, activation='softmax')(x)  # CIFAR-10 has 10 classes

# Build Model
model = models.Model(inputs=inputs, outputs=outputs)

# Compile the Model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Show the Model Summary
model.summary()
# ===========================
#  Data
# ===========================
train_dataset = np.load('cifar10_train.npz')
val_dataset   = np.load('cifar10_val.npz')
test_dataset  = np.load('cifar10_test.npz')

x_train, y_train = train_dataset['x'], train_dataset['y']
x_val,   y_val   = val_dataset['x'], val_dataset['y']
x_test,  y_test  = test_dataset['x'], test_dataset['y']

x_train = x_train / 255.0; x_val = x_val / 255.0; x_test = x_test / 255.0

train_datagen = ImageDataGenerator(rotation_range=15, width_shift_range=0.1,
                                   height_shift_range=0.1, horizontal_flip=True)

#train_datagen = ImageDataGenerator(rotation_range=15, width_shift_range=0.1,
                                   #height_shift_range=0.1, horizontal_flip=True)

val_datagen   = ImageDataGenerator(); test_datagen = ImageDataGenerator()

train_generator      = train_datagen.flow(x_train, y_train, batch_size=128, shuffle=True)
validation_generator = val_datagen.flow(x_val, y_val, batch_size=128, shuffle=False)
test_generator       = test_datagen.flow(x_test, y_test, batch_size=128, shuffle=False)


# ---- number of classes (single source of truth) ----
# Use labels or model output to determine classes
num_classes = int(np.unique(y_train).size)  # or: num_classes = model.output_shape[-1]

# CIFAR-10 names (adjust if you’re not using CIFAR-10)
class_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck'][:num_classes]
# ===========================
#  Train
# ===========================
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
process = psutil.Process(); start_time = time.time()
initial_memory = process.memory_info().rss / (1024 ** 2)

history = model.fit(train_generator, epochs=60, validation_data=validation_generator,
                    callbacks=[reduce_lr])

end_time = time.time(); final_memory = process.memory_info().rss / (1024 ** 2)
training_time = end_time - start_time; memory_used = final_memory - initial_memory
print(f"Training Time: {training_time:.2f} seconds"); print(f"Memory Used: {memory_used:.2f} MB")

# ===========================
#  Curves
# ===========================
train_acc = history.history['accuracy']; val_acc = history.history['val_accuracy']
train_loss = history.history['loss'];    val_loss = history.history['val_loss']

print(f"Max Training Accuracy: {max(train_acc):.4f}, Min Training Accuracy: {min(train_acc):.4f}")
print(f"Max Validation Accuracy: {max(val_acc):.4f}, Min Validation Accuracy: {min(val_acc):.4f}")
print(f"Min Training Loss: {min(train_loss):.4f}, Max Training Loss: {max(train_loss):.4f}")
print(f"Min Validation Loss: {min(val_loss):.4f}, Max Validation Loss: {max(val_loss):.4f}")

plt.figure(figsize=(20, 8))
plt.subplot(1, 2, 1)
plt.plot(train_acc, label='Training Accuracy'); plt.plot(val_acc, label='Validation Accuracy')
plt.title('Accuracy over Epochs'); plt.xlabel('Epochs'); plt.ylabel('Accuracy'); plt.legend(); plt.grid(True)
plt.subplot(1, 2, 2)
plt.plot(train_loss, label='Training Loss'); plt.plot(val_loss, label='Validation Loss')
plt.title('Loss over Epochs'); plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)
plt.tight_layout(); plt.savefig("training_curves.png", dpi=200); plt.show()

# ===========================
#  Evaluate
# ===========================
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Loss: {test_loss:.4f}"); print(f"Test Accuracy: {test_accuracy:.4f}")

predicted_classes = np.argmax(model.predict(test_generator), axis=1)
true_classes = y_test.flatten()

print("\nClassification Report:"); print(classification_report(true_classes, predicted_classes, digits=4))

os.makedirs("interpretability", exist_ok=True)

conf_matrix = confusion_matrix(true_classes, predicted_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Classes'); plt.ylabel('True Classes'); plt.title('Confusion Matrix')
plt.tight_layout(); plt.savefig("interpretability/confusion_matrix.png", dpi=200); plt.show()

precision = precision_score(true_classes, predicted_classes, average='weighted')
recall    = recall_score(true_classes, predicted_classes, average='weighted')
f1        = f1_score(true_classes, predicted_classes, average='weighted')
print(f"Precision (Weighted): {precision:.4f}"); print(f"Recall (Weighted): {recall:.4f}"); print(f"F1-Score (Weighted): {f1:.4f}")


