import os
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Masking, TimeDistributed, Activation, Bidirectional, Lambda, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import gdown

# Function to download files from Google Drive
def download_file_from_google_drive(file_id, dest_path):
    gdown.download(f'https://drive.google.com/uc?id={file_id}', dest_path, quiet=False)

# Download and unzip main dataset
url = 'https://drive.google.com/uc?id=1jyvhdZHn0s5Owkr21k5Ff-c96sIQLtEu'
output = 'all_wav.zip'
download_file_from_google_drive('1jyvhdZHn0s5Owkr21k5Ff-c96sIQLtEu', output)
!unzip -q 'all_wav.zip' -d '/content/all_wav'

url = 'https://drive.google.com/uc?id=1vqvn0F0YYhEFbzLgP9wJ36vyInUnO5b5'
output = 'dataset.csv'
download_file_from_google_drive('1vqvn0F0YYhEFbzLgP9wJ36vyInUnO5b5', output)

# Load dataset
df = pd.read_csv('/content/dataset.csv')

# Update paths for WAV files
df['wav_filename'] = df['wav_filename'].apply(lambda x: x.replace('./all_wav/', '/content/all_wav/all_wav/'))

# Filter dataset to include only existing WAV files
df = df[df['wav_filename'].apply(os.path.isfile)]

# Character map for Persian characters
char_map_str = """
' 0
<SPACE> 1
ا 2
ب 3
پ 4
ت 5
ث 6
ج 7
چ 8
ح 9
خ 10
د 11
ذ 12
ر 13
ز 14
ژ 15
س 16
ش 17
ص 18
ض 19
ط 20
ظ 21
ع 22
غ 23
ف 24
ق 25
ک 26
گ 27
ل 28
م 29
ن 30
و 31
ه 32
ی 33
، 34
؟ 35
"""
char_map = {}
index_map = {}
for line in char_map_str.strip().split('\n'):
    ch, index = line.split()
    char_map[ch] = int(index)
    index_map[int(index)] = ch
index_map[1] = ' '

# Ensure space character is in char_map
char_map[' '] = char_map['<SPACE>']

# Audio processing functions
def load_audio(file_path, sr=16000):
    audio, _ = librosa.load(file_path, sr=sr)
    return audio

def extract_features(audio, n_mfcc=20, sr=16000):
    mfcc_features = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    delta_mfcc = librosa.feature.delta(mfcc_features)
    combined = np.vstack((mfcc_features, delta_mfcc)).T
    return combined

def add_noise(audio, noise_factor=0.005):
    noise = np.random.randn(len(audio))
    augmented_audio = audio + noise_factor * noise
    augmented_audio = augmented_audio.astype(type(audio[0]))
    return augmented_audio

def shift_time(audio, shift_max=0.2):
    shift = np.random.randint(int(shift_max * 16000))
    if np.random.rand() > 0.5:
        shift = -shift
    augmented_audio = np.roll(audio, shift)
    if shift > 0:
        augmented_audio[:shift] = 0
    else:
        augmented_audio[shift:] = 0
    return augmented_audio

# Prepare the dataset
augment_functions = [add_noise, shift_time]
X, y, input_lengths, label_lengths = [], [], [], []

for index, row in df.iterrows():
    audio_path = row['wav_filename']
    audio = load_audio(audio_path)
    
    # Original audio features
    features = extract_features(audio)
    X.append(features)
    input_lengths.append(features.shape[0])
    label = [char_map.get(c, char_map[' ']) for c in row['transcript']]
    y.append(label)
    label_lengths.append(len(label))
    
    # Augmented audio features
    for augment_func in augment_functions:
        augmented_audio = augment_func(audio)
        features = extract_features(augmented_audio)
        X.append(features)
        input_lengths.append(features.shape[0])
        y.append(label)
        label_lengths.append(len(label))

if len(X) == 0 or len(y) == 0:
    raise ValueError("No valid audio files were found. Please check the dataset and the paths.")

# Save augmented data to files
np.save('/content/X_augmented.npy', X)
np.save('/content/y_augmented.npy', y)
np.save('/content/input_lengths_augmented.npy', input_lengths)
np.save('/content/label_lengths_augmented.npy', label_lengths)

# Option to download augmented data from Google Drive
file_urls = {
    'X_augmented.npy': 'https://drive.google.com/uc?id=1-5QQGFBQuL4AO9XTMwb4o7TlTpsoG9M4',
    'y_augmented.npy': 'https://drive.google.com/uc?id=1-7-Oh7Mj2qaVNr8eBAMhvzMubhDoLsSX',
    'input_lengths_augmented.npy': 'https://drive.google.com/uc?id=1-7yhidCIPb2EdKN7ZDweHOf4Q9sNd63a',
    'label_lengths_augmented.npy': 'https://drive.google.com/uc?id=1-77poJQcMc1V5GqnHLazIfjZcvDbvkVi'
}

for file_name, file_url in file_urls.items():
    gdown.download(file_url, f'/content/{file_name}', quiet=False)

# Load the augmented data using memory-mapped files
X = np.load('/content/X_augmented.npy', mmap_mode='r')
y = np.load('/content/y_augmented.npy', mmap_mode='r')
input_lengths = np.load('/content/input_lengths_augmented.npy', mmap_mode='r')
label_lengths = np.load('/content/label_lengths_augmented.npy', mmap_mode='r')

# Define the model with increased LSTM units to 256
input_data = Input(name='the_input', shape=(None, 40))
masking_layer = Masking(mask_value=0.0)(input_data)
bilstm_layer_1 = Bidirectional(LSTM(256, return_sequences=True))(masking_layer)
batch_norm_1 = BatchNormalization()(bilstm_layer_1)
dropout_1 = Dropout(0.3)(batch_norm_1)
bilstm_layer_2 = Bidirectional(LSTM(256, return_sequences=True))(dropout_1)
batch_norm_2 = BatchNormalization()(bilstm_layer_2)
dropout_2 = Dropout(0.3)(batch_norm_2)
time_dense = TimeDistributed(Dense(len(char_map) + 1))(dropout_2)
y_pred = Activation('softmax', name='activation')(time_dense)

# Define the CTC loss function
def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    return tf.keras.backend.ctc_batch_cost(labels, y_pred, input_length, label_length)

# Compile the model with CTC loss
labels = Input(name='the_labels', shape=[None], dtype='float32')
input_length = Input(name='input_length', shape=[1], dtype='int64')
label_length = Input(name='label_length', shape=[1], dtype='int64')
loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])

model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss={'ctc': lambda y_true, y_pred: y_pred})

# Split the data into training and validation sets
X_train, X_val, y_train, y_val, input_length_train, input_length_val, label_length_train, label_length_val = train_test_split(
    X, y, input_lengths, label_lengths, test_size=0.2, random_state=42)

# Data generator
def data_generator(X, y, input_lengths, label_lengths, batch_size=16):
    while True:
        for i in range(0, len(X), batch_size):
            X_batch = X[i:i+batch_size]
            y_batch = y[i:i+batch_size]
            input_lengths_batch = input_lengths[i:i+batch_size]
            label_lengths_batch = label_lengths[i:i+batch_size]
            yield (
                {
                    'the_input': np.array(X_batch),
                    'the_labels': np.array(y_batch),
                    'input_length': np.array(input_lengths_batch),
                    'label_length': np.array(label_lengths_batch)
                },
                {'ctc': np.zeros([len(X_batch)])}
            )

train_gen = data_generator(X_train, y_train, input_length_train, label_length_train, batch_size=16)
val_gen = data_generator(X_val, y_val, input_length_val, label_length_val, batch_size=16)

steps_per_epoch = len(X_train) // 16
validation_steps = len(X_val) // 16

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('/content/asr_best_model.keras', monitor='val_loss', save_best_only=True)

# Train the model
model.fit(train_gen, steps_per_epoch=steps_per_epoch, epochs=30, validation_data=val_gen, validation_steps=validation_steps, callbacks=[early_stopping, model_checkpoint])

# Save the final model
model.save('/content/asr_model.keras')

# Model summary
model.summary()

# Redefine the inference model
inference_model = Model(inputs=input_data, outputs=y_pred)

# Load the weights from the best saved model
inference_model.load_weights('/content/asr_best_model.keras')

# Function to predict on a new sample
def predict_sample(sample_index):
    sample_features = X[sample_index]
    sample_input_length = np.array([sample_features.shape[0]])

    sample_features = np.expand_dims(sample_features, axis=0)
    sample_input_length = np.array([sample_features.shape[1]], dtype=np.int32)

    # Predict with beam search decoding
    preds = inference_model.predict(sample_features)
    decoded_pred = tf.keras.backend.ctc_decode(preds, input_length=sample_input_length, greedy=False, beam_width=20, top_paths=1)[0][0]
    decoded_pred = tf.keras.backend.get_value(decoded_pred)

    # Ensure decoded_pred is a 1D array
    decoded_pred = decoded_pred.flatten()

    # Convert the decoded prediction to text
    predicted_text = ''.join([index_map[i] for i in decoded_pred if i != -1])

    # Actual text from the dataset
    actual_text = df.iloc[sample_index]['transcript']

    print(f"Predicted text: {predicted_text}")
    print(f"Actual text: {actual_text}")

# Test the model on a new sample
predict_sample(25)