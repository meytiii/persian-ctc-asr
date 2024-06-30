
# Persian Automatic Speech Recognition (ASR) using CTC Loss

## Introduction

This project is focused on developing an Automatic Speech Recognition (ASR) system for the Persian language using Connectionist Temporal Classification (CTC) loss. The objective is to build a model that can transcribe spoken Persian into text. This project involves multiple steps including data preprocessing, feature extraction, model building, training, and evaluation.

### Why Persian ASR?

ASR systems have been predominantly developed for languages like English, Mandarin, and Spanish. However, there is a growing need for ASR systems in other languages, including Persian, due to the increasing digital content in these languages. Building a Persian ASR system helps in various applications such as voice assistants, transcription services, and accessibility tools for Persian speakers.

## Dataset

The dataset used for this project includes Persian audio files and their corresponding transcriptions. The audio files are in WAV format, and the transcriptions are provided in a CSV file.

-   **Audio Files**: The audio files are stored in a directory and are named sequentially.
-   **Transcriptions**: The CSV file contains the following columns:
    -   `wav_filename`: Path to the audio file.
    -   `wav_filesize`: Size of the audio file.
    -   `transcript`: The text transcription of the audio.
    -   `confidence_level`: Confidence level of the transcription.

Example of Transcription Data:

-   wav_filename: /content/all_wav/all_wav/Tehran_SayeRoshan0_101.wav
    -   wav_filesize: 83044
    -   transcript: اتفاقاتی که ندیده بودم
    -   confidence_level: 0.927557
-   wav_filename: /content/all_wav/all_wav/Tehran_SayeRoshan0_105.wav
    -   wav_filesize: 54468
    -   transcript: مسجد
    -   confidence_level: 0.927557
-   wav_filename: /content/all_wav/all_wav/Tehran_SayeRoshan0_107.wav
    -   wav_filesize: 136036
    -   transcript: جمع شدن مسلمین برای نمازهای جماعت
    -   confidence_level: 0.864152
-   wav_filename: /content/all_wav/all_wav/Tehran_SayeRoshan0_108.wav
    -   wav_filesize: 106788
    -   transcript: همیشه برای محمدرضا پهلوی
    -   confidence_level: 0.927557
-   wav_filename: /content/all_wav/all_wav/Tehran_SayeRoshan0_109.wav
    -   wav_filesize: 170020
    -   transcript: چه زمانی در کسوت شاه ایران نوکری اجانب را می‌کرد
    -   confidence_level: 0.854824

## Preprocessing and Feature Extraction

### Loading and Cleaning Data

The dataset is loaded, and paths for audio files are updated. The data is filtered to include only existing WAV files.

### Character Mapping

A character map is created to convert Persian characters to numerical indices, which will be used for training the model. This includes Persian alphabets, numerals, and common punctuation marks.

### Audio Feature Extraction

The audio files are processed to extract Mel-Frequency Cepstral Coefficients (MFCC) features. This step converts the audio signals into a form that can be used by the neural network.

## Model Architecture

The model used in this project is a Recurrent Neural Network (RNN) with Bidirectional Long Short-Term Memory (BiLSTM) layers. The architecture is designed to capture the temporal dependencies in the audio data.

### Model Layers

-   **Input Layer**: Takes MFCC features as input.
-   **Masking Layer**: Masks the padding in the input sequences.
-   **Bidirectional LSTM Layers**: Two BiLSTM layers to capture dependencies in both forward and backward directions.
-   **TimeDistributed Dense Layer**: Applies a dense layer to each time step of the sequence.
-   **Activation Layer**: Applies a softmax activation to generate probabilities for each character.

### CTC Loss

Connectionist Temporal Classification (CTC) loss is used to train the model. CTC is suitable for sequence-to-sequence problems where the alignment between input and output is unknown.

## Training

The model is trained using the Adam optimizer with CTC loss. The training data is split into training and validation sets. A custom data generator is used to feed the data to the model in batches.

### Training Process

1.  **Data Preparation**: The data is padded and split into training and validation sets.
2.  **Model Compilation**: The model is compiled with the Adam optimizer and CTC loss.
3.  **Model Training**: The model is trained for a specified number of epochs, with evaluation on the validation set after each epoch.

## Inference

After training, the model is used for inference. A sample audio file is processed, and the model predicts the transcription. The predicted transcription is then compared to the actual transcription.

## Results

The model is evaluated based on the accuracy of the transcriptions. While the model shows promising results, there is room for improvement in terms of accuracy and generalization.

Example Output:

-   Predicted text: تفاقات که ندید بود
-   Actual text: اتفاقاتی که ندیده بودم

## Conclusion

This project demonstrates the process of building an ASR system for the Persian language using CTC loss. The results indicate that with more data and fine-tuning, the model's performance can be significantly improved. This ASR system has the potential to be used in various applications, enhancing accessibility and providing valuable services to Persian-speaking users.

## Future Work

1.  **Data Augmentation**: Use data augmentation techniques to increase the robustness of the model.
2.  **Hyperparameter Tuning**: Experiment with different model architectures and hyperparameters to improve accuracy.
3.  **Additional Features**: Incorporate additional features such as prosody and tone to enhance performance.
4.  **Real-time ASR**: Optimize the model for real-time transcription applications.

## References

1.  Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks
2.  [Librosa: Audio and Music Signal Analysis in Python](https://librosa.org/)
3.  [TensorFlow: An Open Source Machine Learning Framework for Everyone](https://www.tensorflow.org/)

## Acknowledgments

This project is inspired by various works in the field of speech recognition and deep learning. Special thanks to the creators of the datasets and tools used in this project.

## How to Run the Project

1.  **Clone the repository**: `git clone https://github.com/meytiii/persian-ctc-asr`
2.  **Install dependencies**: `pip install -r requirements.txt`
3.  **Download the dataset**: Run the provided script to download and extract the dataset.
4.  **Train the model**: Execute the training script.
5.  **Run inference**: Use the inference script to test the model on sample audio files.

By following these steps, you can set up and run the Persian ASR system on your local machine or in a cloud environment.
