# SMS Spam Detection with NLP
This repository contains a Jupyter Notebook (SMS SPam NLP.ipynb) that demonstrates the process of building a simple SMS spam detection model using Natural Language Processing (NLP) techniques. The code in this notebook utilizes Python libraries such as Pandas, NLTK, and scikit-learn to preprocess text data and train a Multinomial Naive Bayes classifier for SMS spam detection.

# Overview
The primary goal of this project is to classify SMS messages as either spam or not spam (ham). The dataset used for this task is named SMSSpamCollection, and it contains SMS messages labeled as spam or ham. The code in the notebook goes through the following steps:

# Data Loading: The SMS dataset is loaded into a Pandas DataFrame, where each row contains a message and its corresponding label (spam or ham).

# Text Preprocessing: The text data is preprocessed to clean and prepare it for modeling. The following preprocessing steps are performed:

Removing non-alphabetic characters.
Converting text to lowercase.
Tokenization (splitting text into words).
Lemmatization (reducing words to their base form).
Removing stopwords (common words that do not carry significant meaning).
Feature Extraction: The cleaned text data is transformed into numerical features using the CountVectorizer, which converts text into a bag-of-words representation. It creates a matrix where each row represents a document (SMS message) and each column represents a unique word in the corpus.

Train-Test Split: The dataset is split into training and testing sets to evaluate the model's performance. 80% of the data is used for training, and 20% for testing.

Model Building: A Multinomial Naive Bayes classifier is trained on the training data to learn the patterns in the SMS messages.

Model Evaluation: The model's performance is evaluated using a confusion matrix and accuracy score on the testing data.

# Usage
To replicate the SMS spam detection process, follow these steps:

Clone this repository to your local machine.

Ensure you have the required Python libraries installed. You can do this by running the following commands:

python
Copy code
pip install pandas nltk scikit-learn
Open and run the SMS SPam NLP.ipynb Jupyter Notebook. It will guide you through the entire process, explaining each step along the way.

Feel free to modify the code, dataset, or parameters as needed for your specific use case or dataset.

# Dataset
The SMS dataset used in this project is available in the SMSSpamCollection file. It contains SMS messages labeled as either "spam" or "ham" (not spam).



# License
This project is open-source and available under the MIT License. Feel free to use, modify, and distribute it as needed. If you find it helpful, please give credit by referencing this repository.
