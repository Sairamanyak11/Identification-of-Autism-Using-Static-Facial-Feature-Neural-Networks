import numpy as np
import cv2
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from tensorflow.keras.applications import ResNet50, Xception
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import MaxPooling2D, Dense, Flatten, Conv2D
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# --- Configuration ---
labels = ['Autistic', 'Non_Autistic']
img_size = (80, 80)
model_dir = 'model'

# Ensure model directory exists
os.makedirs(model_dir, exist_ok=True)

def getID(name):
    return labels.index(name) if name in labels else 0

def load_and_preprocess_dataset(dataset_dir, img_size=(80,80)):
    X, Y = [], []
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')) and file != 'Thumbs.db':
                img_path = os.path.join(root, file)
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Warning: Skipping unreadable image: {img_path}")
                    continue
                img = cv2.resize(img, img_size)
                im2arr = np.array(img)
                label = getID(os.path.basename(root))
                X.append(im2arr)
                Y.append(label)
                print(f"Loaded {file} with label {label}")
    if not X:
        raise RuntimeError("No images loaded — check dataset path and folder structure.")
    X = np.array(X, dtype='float32') / 255.0
    Y = np.array(Y)
    Y = to_categorical(Y)
    return X, Y

def split_dataset(X, Y, test_size=0.2):
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    return train_test_split(X, Y, test_size=test_size)

def build_resnet_model(input_shape, num_classes):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    model = Sequential([
        base_model,
        Conv2D(32, (1,1), activation='relu'),
        MaxPooling2D((1,1)),
        Flatten(),
        Dense(256, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def build_xception_model(input_shape, num_classes):
    base_model = Xception(weights='imagenet', include_top=False, input_shape=input_shape)
    model = Sequential([
        base_model,
        Conv2D(32, (3,3), strides=(3,3), activation='relu', padding='same'),
        MaxPooling2D((2,2), padding='same'),
        Flatten(),
        Dense(256, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, X_train, y_train, X_val, y_val, model_path, epochs=5, batch_size=16):
    if not os.path.exists(model_path):
        checkpoint = ModelCheckpoint(model_path, save_best_only=True, verbose=1)
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                            epochs=epochs, batch_size=batch_size, callbacks=[checkpoint], verbose=1)
        # Save training history
        with open(model_path.replace('.keras', '_history.pkl'), 'wb') as f:
            pickle.dump(history.history, f)
    else:
        model = load_model(model_path)
    return model

def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    y_pred = np.argmax(preds, axis=1)
    y_true = np.argmax(y_test, axis=1)

    acc = accuracy_score(y_true, y_pred) * 100
    prec = precision_score(y_true, y_pred, average='macro') * 100
    rec = recall_score(y_true, y_pred, average='macro') * 100
    f1 = f1_score(y_true, y_pred, average='macro') * 100

    print("\n--- Evaluation Metrics ---")
    print(f"Accuracy:  {acc:.2f}%")
    print(f"Precision: {prec:.2f}%")
    print(f"Recall:    {rec:.2f}%")
    print(f"F1 Score:  {f1:.2f}%")

    conf_matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    return acc, prec, rec, f1

def predict_image(model, image_path, img_size=(80,80)):
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return None

    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, img_size)
    img_array = np.array(img_resized, dtype='float32') / 255.0
    img_array = img_array.reshape(1, *img_size, 3)

    preds = model.predict(img_array)
    pred_label = labels[np.argmax(preds)]

    print(f"Prediction: {pred_label}")
    return pred_label

# --- Main Execution ---
if __name__ == "__main__":
    dataset_dir = "C:/Users/dheer/OneDrive/Desktop/sathwika/dataset"  # 🔁 Replace with your actual dataset path if different
    print("--- Loading and Preprocessing Dataset ---")
    X, Y = load_and_preprocess_dataset(dataset_dir, img_size=img_size)
    X_train, X_test, y_train, y_test = split_dataset(X, Y)

    print("\n--- Training ResNet50 ---")
    resnet_model = build_resnet_model(input_shape=(80,80,3), num_classes=y_train.shape[1])
    resnet_model = train_model(resnet_model, X_train, y_train, X_test, y_test,
                               model_path=os.path.join(model_dir, "resnet_model.keras"))

    print("\n--- Evaluating ResNet50 ---")
    evaluate_model(resnet_model, X_test, y_test)

    print("\n--- Training Xception ---")
    xception_model = build_xception_model(input_shape=(80,80,3), num_classes=y_train.shape[1])
    xception_model = train_model(xception_model, X_train, y_train, X_test, y_test,
                                 model_path=os.path.join(model_dir, "xception_model.keras"))

    print("\n--- Evaluating Xception ---")
    evaluate_model(xception_model, X_test, y_test)

    # Optional: Predict an image
    # image_path = "path_to_image.jpg"  # 🔁 Replace with test image path
    # predict_image(resnet_model, image_path)
