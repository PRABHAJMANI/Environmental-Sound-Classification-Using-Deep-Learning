import os
import numpy as np
import pandas as pd
import librosa
import librosa.display
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from datetime import datetime

# Paths
AUDIO_DATASET_PATH = "UrbanSound8K/audio/"
METADATA_PATH = "UrbanSound8K/metadata/UrbanSound8K.csv"
SAVED_MODEL_PATH = "saved_models/audio_classification.hdf5"

# Function to extract MFCC features
def features_extractor(file_name):
    audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
    return mfccs_scaled_features

# Load metadata
metadata = pd.read_csv(METADATA_PATH)

# Extract features for all audio files
extracted_features = []
for index_num, row in metadata.iterrows():
    file_name = os.path.join(os.path.abspath(AUDIO_DATASET_PATH),
                             'fold'+str(row["fold"])+'/',
                             str(row["slice_file_name"]))
    final_class_labels = row["class"]
    try:
        data = features_extractor(file_name)
        extracted_features.append([data, final_class_labels])
    except Exception as e:
        print(f"Error processing file {file_name}: {e}")

# Convert to DataFrame
extracted_features_df = pd.DataFrame(extracted_features, columns=['feature', 'class'])

# Split data into features and labels
X = np.array(extracted_features_df['feature'].tolist())
y = np.array(extracted_features_df['class'].tolist())

# Encode labels
y = to_categorical(LabelEncoder().fit_transform(y))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model definition
num_labels = y.shape[1]
model = Sequential()
model.add(Dense(100, input_shape=(40,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(200))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_labels))
model.add(Activation('softmax'))

# Compile model
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
model.summary()

# Training
num_epochs = 100
num_batch_size = 32
checkpointer = ModelCheckpoint(filepath=SAVED_MODEL_PATH, verbose=1, save_best_only=True)
start = datetime.now()

history = model.fit(X_train, y_train, batch_size=num_batch_size, epochs=num_epochs,
                    validation_data=(X_test, y_test), callbacks=[checkpointer], verbose=1)

# Training duration
duration = datetime.now() - start
print(f"Training completed in time: {duration}")

# Model evaluation
test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test accuracy: {test_accuracy[1]}")
