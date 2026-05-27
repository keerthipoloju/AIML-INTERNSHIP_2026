import argparse, json, os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

CLASS_NAMES = ["angry","disgust","fear","happy","sad","surprise","neutral"]

def parse_pixels(pixels_str):
    vals = np.fromstring(pixels_str, sep=' ', dtype=np.float32)
    img = vals.reshape(48, 48, 1) / 255.0
    return img

def load_fer_csv(path):
    df = pd.read_csv(path)
    # Expected columns: emotion, pixels, Usage
    df = df[df['emotion'].isin(range(7))].copy()
    df['label'] = df['emotion'].astype(int)
    return df

def df_to_dataset(df, batch_size=128, augment=False, shuffle=True):
    X = np.stack(df['pixels'].apply(parse_pixels).values)
    y = tf.keras.utils.to_categorical(df['label'].values, num_classes=7)

    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(df), reshuffle_each_iteration=True)

    if augment:
        aug = tf.keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.08),
            layers.RandomZoom(0.08),
        ])
        ds = ds.map(lambda x, y: (aug(x), y), num_parallel_calls=tf.data.AUTOTUNE)

    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

def build_model():
    inputs = layers.Input(shape=(48,48,1))
    x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(7, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def main(args):
    csv_path = args.csv
    if not csv_path:
        csv_path = os.path.join("data", "fer2013", "fer2013.csv")
    print(f"Loading FER-2013 from: {csv_path}")
    df = load_fer_csv(csv_path)

    train_df = df[df['Usage'].str.contains('Training', case=False, na=False)].copy()
    val_df   = df[df['Usage'].str.contains('PublicTest', case=False, na=False)].copy()
    test_df  = df[df['Usage'].str.contains('PrivateTest', case=False, na=False)].copy()

    # Keep only the two needed columns
    for subset in (train_df, val_df, test_df):
        subset['pixels'] = subset['pixels'].astype(str)

    train_ds = df_to_dataset(train_df, batch_size=args.batch_size, augment=True, shuffle=True)
    val_ds   = df_to_dataset(val_df, batch_size=args.batch_size, augment=False, shuffle=False)
    test_ds  = df_to_dataset(test_df, batch_size=args.batch_size, augment=False, shuffle=False)

    model = build_model()
    model.summary()

    ckpt_path = os.path.join("models", "best_fer_cnn.keras")
    cb = [
        callbacks.ModelCheckpoint(ckpt_path, monitor='val_accuracy', save_best_only=True, mode='max'),
        callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor='val_accuracy', mode='max')
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=cb
    )

    # Evaluate
    loss, acc = model.evaluate(test_ds, verbose=0)
    print(f"Test accuracy: {acc:.4f}")

    # Save final model and class names
    final_path = os.path.join("models", "fer_cnn.keras")
    model.save(final_path)
    with open(os.path.join("models", "class_names.json"), "w") as f:
        json.dump(CLASS_NAMES, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=None, help="Path to fer2013.csv")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()
    main(args)
