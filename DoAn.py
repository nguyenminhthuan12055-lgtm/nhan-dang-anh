import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import pathlib
import os
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Tắt cảnh báo hệ thống
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# --- 1. CẤU HÌNH ---
DATASET_PATH = pathlib.Path(r"C:\Users\ACER\OneDrive\Documents\raw-img\Animal-10")
TEST_FOLDER = r"C:\Users\ACER\OneDrive\Documents\test_images"
MODEL_FILE = 'animal_model_v5_final.keras'
IMG_SIZE = (160, 160)
BATCH_SIZE = 16
INITIAL_EPOCHS = 50
FINE_TUNE_EPOCHS = 20 # Tổng cộng 70 Epoch

# --- 2. DATA AUGMENTATION (NÂNG CẤP ĐỂ CHỐNG OVERFITTING) ---
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.2),
    layers.RandomTranslation(0.1, 0.1), # Giúp mô hình linh hoạt hơn với vị trí
])

def load_data():
    print("--- [BƯỚC 1] TẢI DỮ LIỆU ---")
    train_raw = tf.keras.utils.image_dataset_from_directory(
        DATASET_PATH, validation_split=0.2, subset="training", seed=123,
        image_size=IMG_SIZE, batch_size=BATCH_SIZE)
    val_raw = tf.keras.utils.image_dataset_from_directory(
        DATASET_PATH, validation_split=0.2, subset="validation", seed=123,
        image_size=IMG_SIZE, batch_size=BATCH_SIZE)
    
    class_names = train_raw.class_names
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_raw.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_raw.cache().prefetch(buffer_size=AUTOTUNE)
    return train_ds, val_ds, class_names

def build_model(num_classes):
    print("--- [BƯỚC 2] KHỞI TẠO MOBILENETV2 (CHỐNG OVERFITTING) ---")
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(160, 160, 3), include_top=False, weights='imagenet')
    base_model.trainable = False

    model = keras.Sequential([
        layers.Input(shape=(160, 160, 3)),
        data_augmentation,
        layers.Rescaling(1./127.5, offset=-1),
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5), # Tăng Dropout để giảm học vẹt
        # Thêm Regularizer L2 để kiểm soát trọng số
        layers.Dense(num_classes, activation='softmax', kernel_regularizer=regularizers.l2(0.001))
    ])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model, base_model

def save_training_history(history, history_fine):
    print("\n--- [BƯỚC 3.5] HIỂN THỊ BIỂU ĐỒ ACCURACY & LOSS ---")
    acc = history.history['accuracy'] + history_fine.history['accuracy']
    val_acc = history.history['val_accuracy'] + history_fine.history['val_accuracy']
    loss = history.history['loss'] + history_fine.history['loss']
    val_loss = history.history['val_loss'] + history_fine.history['val_loss']

    plt.figure(figsize=(12, 10))
    
    # Biểu đồ Accuracy
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy', color='blue')
    plt.plot(val_acc, label='Validation Accuracy', color='orange')
    plt.axvline(x=INITIAL_EPOCHS-1, color='red', linestyle='--', label='Fine-tuning Start')
    plt.title('Độ chính xác qua 70 Epoch')
    plt.legend(loc='lower right')
    plt.grid(True)

    # Biểu đồ Loss
    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss', color='blue')
    plt.plot(val_loss, label='Validation Loss', color='orange')
    plt.axvline(x=INITIAL_EPOCHS-1, color='red', linestyle='--', label='Fine-tuning Start')
    plt.title('Giá trị Mất mát (Loss)')
    plt.legend(loc='upper right')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def show_detailed_report(model, val_ds, class_names):
    print("\n--- [BƯỚC 5.5] TEST RESULTS (BÁO CÁO CHI TIẾT GIỐNG HÌNH) ---")
    y_true, y_pred = [], []
    for images, labels in val_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(preds, axis=1))
    
    # In báo cáo chi tiết Precision, Recall, F1-score
    print(classification_report(y_true, y_pred, target_names=class_names))

def plot_confusion_matrix(model, val_ds, class_names):
    print("\n--- [BƯỚC 5] TẠO MA TRẬN NHẦM LẪN ---")
    y_true, y_pred = [], []
    for images, labels in val_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(preds, axis=1))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Dự đoán')
    plt.ylabel('Thực tế')
    plt.title('Confusion Matrix')
    plt.show()

def main():
    train_ds, val_ds, class_names = load_data()
    
    if os.path.exists(MODEL_FILE):
        print(f"✅ Đang tải mô hình hiện có...")
        model = tf.keras.models.load_model(MODEL_FILE)
        base_model = model.layers[2] 
    else:
        model, base_model = build_model(len(class_names))

    print(f"\n--- GIAI ĐOẠN 1: HUẤN LUYỆN (50 EPOCHS) ---")
    history = model.fit(train_ds, validation_data=val_ds, epochs=INITIAL_EPOCHS)

    print(f"\n--- GIAI ĐOẠN 2: TINH CHỈNH (20 EPOCHS) ---")
    base_model.trainable = True
    # Đóng băng 100 lớp đầu để giữ lại các đặc trưng cơ bản của MobileNetV2
    for layer in base_model.layers[:100]: layer.trainable = False
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    history_fine = model.fit(train_ds, validation_data=val_ds, 
                            epochs=INITIAL_EPOCHS + FINE_TUNE_EPOCHS,
                            initial_epoch=history.epoch[-1])
    
    model.save(MODEL_FILE)
    save_training_history(history, history_fine)
    plot_confusion_matrix(model, val_ds, class_names)
    show_detailed_report(model, val_ds, class_names)

if __name__ == "__main__":
    main()