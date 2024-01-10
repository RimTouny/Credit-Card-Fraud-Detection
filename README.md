# **Credit Card Fraud Detection**
Focused on advancing credit card fraud detection, this project employs machine learning algorithms, including neural networks and decision trees, to enhance fraud prevention in the banking sector using [Fraud Dataset](https://github.com/RimTouny/Credit-Card-Fraud-Detection/files/13895290/Fraud.Dataset.csv). It serves as the final project for a Data Science course at the University of Ottawa in 2023.

- Required libraries: scikit-learn, pandas, matplotlib.
- Execute cells in a Jupyter Notebook environment.
- The uploaded code has been executed and tested successfully within the [Google Colab](https://colab.google/) environment.


## Binary-class classification problem
Task is to classify whether a credit card transaction is fraudulent or not based on various features, enhancing fraud detection in financial transactions: 1 /0

### Independent Variables: include transaction details, credit card information, merchant data, billing address, cardholder demographics, occupation, and transaction timestamps.
     
### Target variable:
   +	'is_fraud': Binary variable indicating whether the transaction is fraudulent (1) or not (0).

## **Key Tasks Undertaken**

1. **Set-up**
 - Loaded and displayed the dataset using Pandas.
 - Generated a profile report using the Pandas Profiling library.
 - Checked the dataset's basic information using `info()` and unique values using `nunique()`.
 - Checked the Data Balance
   ![image](https://github.com/RimTouny/Credit-Card-Fraud-Detection/assets/48333870/5af015c0-0622-4fa0-baff-237dd3ddb538)
- Average values of different features for fraudulent and non-fraudulent transactions
  ![image](https://github.com/RimTouny/Credit-Card-Fraud-Detection/assets/48333870/e5e4e284-be33-4a22-afb6-22022605662b)


2. **Data Pre-processing**
  - Missing and Duplicate Data: Checked for missing values and duplicate rows.
  - Feature Engineering: Calculated the age of credit card holders based on transaction and birth dates and isualized the age distribution in fraudulent and non-fraudulent transactions.
    ![merge_from_ofoct](https://github.com/RimTouny/Credit-Card-Fraud-Detection/assets/48333870/a80968a0-3909-4580-83b9-5e8581ba0913)
  - Feature Selection: Dropped unnecessary columns ans calculated correlation matrix and visualized it using a heatmap.
  - Dealing with Outliers.
  - Encoding Categorical Variables:using Label Encoder after identifing categorical and numerical features.
  - Feature Scaling:using Standard Scaler.
  - Dealing with Imbalanced Data: Dealing with Imbalanced Data.
  - Dimensionality Reduction and Data Visualization: using PCA & t-SNE for  dimensionality reduction and visualization.(Not Applied in Modeling phase)   
    [merge_from_ofoct](https://github.com/RimTouny/Credit-Card-Fraud-Detection/assets/48333870/d083ceef-f652-4f6b-b8e7-47328c2fc9c8)


3. **Data Modeling**
    -A diverse set of classifiers, including SVM, Random Forest, Naive Bayes variants, KNN, XGBoost, SGD, Logistic Regression, Decision Tree, AdaBoost, and CatBoost, are employed for predict fraud .
   
5. **Evaluation**
   - Using cross-validation, confusion matrices for each classifier for training and testing.
   - Calculated accuracy, precision, recall, and F1 score for each classifier for training and testing.
   - Compare the result with /without using PCA-Dimensionality Reduction.
     + Applying PCA-Dimensionality Reduction.
       ![image](https://github.com/RimTouny/Credit-Card-Fraud-Detection/assets/48333870/41818a1d-4022-4b81-bfb6-e8ea88adad40)

     + Without applying PCA-Dimensionality Reduction. (Complete the work Without applying PCA)
       ![image](https://github.com/RimTouny/Credit-Card-Fraud-Detection/assets/48333870/6101fed9-87c1-4a41-9a44-694674f7b513)

6. **Champion Model**: XGB Extreme X Gradient Boosting	
```python
   Cross_validation Accuracy for XGB Extreme X Gradient Boosting :[0.99937487 0.99874974 0.99916649 0.99749948 0.99854136 0.99833299 0.99812461 0.99874974 0.99895812 0.99874974]
 ```
  ![merge_from_ofoct](https://github.com/RimTouny/Credit-Card-Fraud-Detection/assets/48333870/f750ecb5-10e7-4364-8b6c-7fea07c97149)

6. **Supervised Deep Learning Algorithms**:
  - Neural Network (NN) Model
      + Data Splitting: Split the training data into training and validation sets.
      + Model Architecture: Created a Sequential neural network model with three hidden layers and an output layer.
      + Compilation: Compiled the model using the RMSprop optimizer and binary crossentropy loss function.
      + Training: Trained the NN model for 100 epochs on the training data.

      - **Evaluation and Results**
          + Predictions: Obtained predictions on the validation set.
          + Classification Report: Generated a classification report, showing precision, recall, and F1-score.
          + Precision Analysis: Examined precision, which turned out to be 1, indicating a strong positive predictive value.
          + Model Evaluation: Evaluated the model on the test set, reshaping predictions and calculating accuracy.

  - Convolutional Neural Network (CNN) Model
      + Data Reshaping: Reshaped the data into 3D as CNN requires 3D input.
      + CNN Architecture: Developed a CNN model with convolutional and pooling layers, aiming to capture spatial features.
      + Model Compilation: Compiled the CNN model using the Adam optimizer and binary crossentropy loss.
      + Training: Trained the CNN model for 45 epochs on the training data.
          ![merge_from_ofoct (1)](https://github.com/RimTouny/Credit-Card-Fraud-Detection/assets/48333870/878e96d6-98f2-48c6-935a-4c4e4c5f096d)

      - **Evaluation and Visualization**
          + Learning Curve: Plotted the learning curve to visualize the training and validation accuracy/loss over epochs.
          +  Max Pooling Enhancement: Modified the CNN architecture by introducing max pooling layers to improve efficiency.
                 ![merge_from_ofoct](https://github.com/RimTouny/Credit-Card-Fraud-Detection/assets/48333870/277a7f98-d80c-4576-94f3-747b9fbb285a)
          +  Final Evaluation: Evaluated the final CNN model on the test set and visualized the confusion matrix.
          + Classification Report:** Displayed a comprehensive classification report with F1-score for each class.



