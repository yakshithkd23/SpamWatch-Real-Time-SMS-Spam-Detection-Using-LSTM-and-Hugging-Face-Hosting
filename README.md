# Spam Message Classification with LSTM  
This repository contains a spam message classification project using a Long Short-Term Memory (LSTM) model. The project aims to classify text messages into two categories: **Spam** or **Ham** (non-spam). The model is built using the TensorFlow and Keras libraries, along with several data preprocessing techniques.

## Sections of the Project

### 1. Data Collection
In this section, we load the dataset for training and testing the model. The dataset consists of labeled SMS messages, where each message is labeled as either 'spam' or 'ham'. The dataset is typically in CSV format, containing two columns:
- **Label**: Represents whether the message is spam (1) or ham (0).
- **Message**: The text content of the message.

The dataset is loaded into a pandas DataFrame, and column names are adjusted for easier access.

### 2. Data Preprocessing
Data preprocessing is crucial to prepare the raw text data for machine learning. In this project, the following steps are applied:
- **Lowercasing**: The text is converted to lowercase to ensure uniformity.
- **Removing special characters**: Non-alphanumeric characters (e.g., punctuation, symbols) are removed.
- **Removing digits**: All numeric values are removed from the text.
- **Tokenization**: The text is converted into tokens (individual words).
- **Padding**: Sequences of text are padded to ensure a uniform input length for the model.

The preprocessed data is then split into training and testing sets for model development.

### 3. Model Development
In this section, we build a spam classification model using a **Long Short-Term Memory (LSTM)** network. The LSTM is a type of Recurrent Neural Network (RNN) that is particularly suited for sequential data like text. The model architecture consists of:
- **Embedding Layer**: This layer converts the text into dense vectors of fixed size.
- **LSTM Layer**: The core of the model, responsible for learning patterns in the text sequences.
- **Dense Layer**: A fully connected layer that produces the output, classifying messages as spam or ham.

The model is compiled with the **Adam** optimizer and **binary cross-entropy** loss function, which is suitable for binary classification tasks.

### 4. Training and Testing
The model is trained on the training dataset using the `fit` method, and its performance is evaluated using the testing dataset. During training, we monitor the model's loss and accuracy. 

- **Training**: The model learns from the data, adjusting its weights to minimize the loss function.
- **Testing**: After training, the model is evaluated on the test data to check its accuracy.

### 5. Model Performance
After training, the model's performance is evaluated using various metrics:
- **Accuracy**: The percentage of correct classifications (both spam and ham).
- **Loss**: A measure of how well the model is performing; lower values are better.
- **Confusion Matrix**: A matrix that shows the performance of the classification model by comparing predicted and actual labels.

The performance of the model can be plotted using various graphs, such as loss curves and accuracy curves, to visualize the training process and evaluate the model's behavior over epochs.
y:
reference: 
   http://doi.one/10.1729/Journal.44657

