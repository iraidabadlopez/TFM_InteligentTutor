import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import random
import pickle
import os
import pandas as pd

tf.random.set_seed(42)
random.seed(42)


#########################################################################################

augment_layer = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomContrast(0.2),
    ])

def augment_if_minority(image, label):
    clase_min = 2 # colocarSoporteSup
    
    image = tf.cond(
        tf.equal(label, clase_min),
        lambda: augment_layer(image, training=True),
        lambda: image
    )
    return image, label

#########################################################################################

def save_history(history, nombre_archivo):
    with open(nombre_archivo, "wb") as f:
        pickle.dump(history.history, f)

###############################################################################

def load_history(nombre_archivo):
    if os.path.exists(nombre_archivo):
        with open(nombre_archivo, "rb") as f:
            loaded_history = pickle.load(f)
        print(f"Historial cargado desde {nombre_archivo}.")
        return loaded_history
    else:
        print(f"No se encontró el archivo {nombre_archivo}.")
        return None

#########################################################################################

def get_dataset(train_paths, train_labels, val_paths, val_labels, test_paths, test_labels, batch_size, norm_method, augment_minority=True):
    
    # Crear dataset de entrenamiento
    train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))

    if augment_minority:
        train_ds = (train_ds
                    .shuffle(len(train_paths)) # Shuffle de rutas
                    .map(lambda x, y: load_image_tf(x, y, method=norm_method), num_parallel_calls=tf.data.AUTOTUNE) # Carga paralela
                    .map(augment_if_minority, num_parallel_calls=tf.data.AUTOTUNE) # Aumentar clase minoritaria
                    .batch(batch_size)
                    .prefetch(tf.data.AUTOTUNE))
    else:
        train_ds = (train_ds
                    .shuffle(len(train_paths)) # Shuffle de rutas
                    .map(lambda x, y: load_image_tf(x, y, method=norm_method), num_parallel_calls=tf.data.AUTOTUNE) # Carga paralela
                    .batch(batch_size)
                    .prefetch(tf.data.AUTOTUNE))

    # Crear dataset de validación
    val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
    val_ds = (val_ds
            .map(lambda x, y: load_image_tf(x, y, method=norm_method), num_parallel_calls=tf.data.AUTOTUNE)
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE))

    # Crear dataset de test
    test_ds = tf.data.Dataset.from_tensor_slices((test_paths, test_labels))
    test_ds = (test_ds
            .map(lambda x, y: load_image_tf(x, y, method=norm_method), num_parallel_calls=tf.data.AUTOTUNE)
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE))
    
    return train_ds, val_ds, test_ds

#########################################################################################

def get_dataset_ts(train_paths, train_labels, val_paths, val_labels, test_paths, test_labels, batch_size, norm_method):
    
    # Crear dataset de entrenamiento
    train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    train_ds = (train_ds
                .map(lambda x, y: load_image_tf(x, y, method=norm_method), num_parallel_calls=tf.data.AUTOTUNE) # Carga paralela
                .batch(batch_size)
                .prefetch(tf.data.AUTOTUNE))

    # Crear dataset de validación
    val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
    val_ds = (val_ds
            .map(lambda x, y: load_image_tf(x, y, method=norm_method), num_parallel_calls=tf.data.AUTOTUNE)
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE))

    # Crear dataset de test
    test_ds = tf.data.Dataset.from_tensor_slices((test_paths, test_labels))
    test_ds = (test_ds
            .map(lambda x, y: load_image_tf(x, y, method=norm_method), num_parallel_calls=tf.data.AUTOTUNE)
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE))
    
    return train_ds, val_ds, test_ds

#########################################################################################

def get_paths_and_labels(df, test_size=0.2, split=True):
    encoder = LabelEncoder()
    df_valid = df[df['full_path'].apply(os.path.exists)].copy()
    
    df_valid['label_encoded'] = encoder.fit_transform(df_valid['label'])
    
    paths = df_valid['full_path'].values
    labels = df_valid['label_encoded'].values

    if split:
    
        # train test (80/20)
        train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
            paths, labels, test_size=test_size, random_state=42, stratify=labels
        )

        # train val (80/20)
        train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths, train_val_labels, test_size=0.2, random_state=42, stratify=train_val_labels
        )

        return (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels), encoder
    
    else:
        return (paths, labels), encoder

#########################################################################################

def load_image_tf(path, label, method):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=3)

    img = tf.cast(img, tf.float32)

    if method is not None:
        if method == "0_1":
            img = tf.cast(img, tf.float32) / 255.0 
        elif method == "neg1_1":
            img = (tf.cast(img, tf.float32) / 127.5) - 1.0
            
    return img, label

def load_img_ts(path, size=(256, 256)):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, size)
    return img / 255.0