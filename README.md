# Audio Classification

This project focuses on classifying environmental sounds using **machine learning** and **deep learning** techniques. The UrbanSound8K dataset is used for training and testing, with features extracted from audio samples to build a robust classification model.

## 🚀 Objective
To classify environmental sounds (e.g., dog bark, drilling, etc.) into predefined categories using **audio processing** and a **deep learning model**.

## 📂 Dataset
The project uses the **[UrbanSound8K dataset](https://urbansounddataset.weebly.com/urbansound8k.html)**, which contains labeled audio files organized into 10 classes, including:
- Air conditioner
- Car horn
- Children playing
- Dog bark
- Drilling

The dataset includes metadata to help locate files and their corresponding class labels.

## 🔑 Key Features
- **Audio Preprocessing**: Audio files are loaded and visualized using `librosa` and `matplotlib`. Features are extracted using **Mel-Frequency Cepstral Coefficients (MFCC)**.
- **Feature Engineering**: Compact feature representations are created by scaling and averaging MFCC features.
- **Deep Learning Model**: Built using TensorFlow/Keras, the model includes:
  - Input and hidden layers with ReLU activations and Dropout for regularization.
  - An output layer with softmax activation for multiclass classification.
- **Evaluation and Prediction**: The trained model is evaluated on test data and used to predict classes for unseen audio files.

## ⚙️ Steps Involved

### 1. **Data Preprocessing**
- Load audio files using `librosa` and `scipy`.
- Visualize waveforms to understand audio signals.
- Extract MFCC features for each audio file.

### 2. **Feature Engineering**
- Extract MFCCs and scale features to summarize frequency and time characteristics.
- Store extracted features and corresponding class labels in a Pandas DataFrame.

### 3. **Train-Test Split**
- Split data into training (80%) and testing (20%) sets.
- Encode class labels into one-hot vectors using `LabelEncoder` and `to_categorical`.

### 4. **Model Creation**
- Build a neural network with the following layers:
  - Input layer: 100 units with ReLU activation.
  - Two hidden layers: 200 and 100 units with ReLU activations and 50% dropout.
  - Output layer: Softmax activation for classification.
- Compile the model with categorical crossentropy loss and Adam optimizer.

### 5. **Training**
- Train the model for 100 epochs with a batch size of 32.
- Use validation data to monitor performance and save the best model using `ModelCheckpoint`.

### 6. **Evaluation and Prediction**
- Evaluate the model's accuracy on the test dataset.
- Preprocess new audio files, extract MFCC features, and predict their classes.

## 🛠️ Libraries and Tools
- **Audio Processing**: `librosa`, `scipy`
- **Data Handling**: `pandas`, `numpy`
- **Visualization**: `matplotlib`
- **Deep Learning**: `TensorFlow`, `Keras`
- **Utilities**: `os`, `tqdm`

## 📊 Results
- The trained model achieves high accuracy in classifying environmental sounds.
- Successfully predicts classes for unseen audio files using the same preprocessing pipeline.

## 📌 How to Run
1. Clone the repository:
   ```bash
   git clone <repository-url>
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download and extract the **UrbanSound8K dataset** into the project directory.
4. Run the training script to preprocess data, train the model, and evaluate performance.
   ```bash
   python train_audio_classification.py
   ```

## 📂 Directory Structure
```
.
├── UrbanSound8K/            # Dataset
├── saved_models/            # Saved models
├── train_audio_classification.py  # Main script
├── requirements.txt         # Dependencies
└── README.md                # Project description
```

## 💡 Future Improvements
- Use pre-trained audio models like **YAMNet** or **OpenL3** for feature extraction.
- Implement data augmentation techniques to improve model generalization.
- Deploy the model as a web service for real-time audio classification.

