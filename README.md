# Customer Churn Prediction Using Artificial Neural Network

# Project Description
This project aims to predict customer churn using an Artificial Neural Network (ANN) implemented in Python. Customer churn prediction is essential for businesses to identify customers who are likely to leave the service, enabling them to take proactive measures to retain those customers. The model is trained on a dataset of customer information, including various demographic, account, and transactional features, to classify whether a customer will churn or not.

# Why is this Useful?
Customer churn is a significant problem for many businesses, particularly those in highly competitive industries. By accurately predicting which customers are likely to leave, businesses can implement targeted retention strategies, improving customer satisfaction and reducing turnover. This can lead to increased revenue and reduced costs associated with acquiring new customers to replace those who have left.

# Libraries Used and Their Purpose

**1. NumPy**
- Description: A fundamental package for scientific computing in Python.
- Usage: Used for efficient numerical operations on large datasets.
  
**2. Pandas**
- Description: A powerful data manipulation and analysis library.
- Usage: Used for loading, preprocessing, and handling the dataset.
  
**3. Matplotlib**
- Description: A plotting library for creating static, animated, and interactive visualizations.
- Usage: Although not explicitly used in the current script, it is often employed for visualizing data distributions and model performance.
  
**4. Scikit-learn**
- Description: A machine learning library providing simple and efficient tools for data analysis and modeling.
- Usage:
- LabelEncoder: Used for encoding categorical features into numerical values.
- OneHotEncoder: Used for one-hot encoding categorical features.
- ColumnTransformer: Applied to preprocess specific columns of the dataset.
- train_test_split: Splits the dataset into training and testing sets.
- StandardScaler: Standardizes features by removing the mean and scaling to unit variance.
- confusion_matrix: Evaluates the performance of the classification model.
  
**5. Keras**
- Description: An open-source neural network library that runs on top of TensorFlow.
- Usage:
- Sequential: Used to initialize the neural network.
- Dense: Used to add fully connected layers to the neural network.
- compile: Compiles the model with specified optimizer and loss functions.
- fit: Trains the model on the training data.
- predict: Generates predictions on the test data.

# Project Workflow

**Data Loading and Preprocessing:**

- Load the dataset using Pandas.
- Encode categorical features using LabelEncoder and OneHotEncoder.
- Split the dataset into training and testing sets.
- Standardize the features using StandardScaler.

**Model Building:**
- Initialize the ANN using Keras Sequential.
- Add input and hidden layers with ReLU activation.
- Add the output layer with sigmoid activation for binary classification.
- Compile the model with Adam optimizer and binary cross-entropy loss function.

**Model Training:**
- Train the model on the training set using the fit method.

**Evaluation:**
- Predict the test set results.
- Evaluate the model performance using a confusion matrix.

# Conclusion
This project demonstrates how to build and train an ANN for predicting customer churn. By leveraging various Python libraries, we can preprocess data, build robust models, and evaluate their performance effectively. This approach provides valuable insights into customer behavior and helps businesses make informed decisions to improve customer retention.
