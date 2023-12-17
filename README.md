# Loan Classification Model

## Overview
This repository contains code for a loan classification model that predicts whether a loan will be approved ('Y') or not ('N') based on various features. The code includes preprocessing steps, model training using k-Nearest Neighbors (k-NN), Support Vector Machine (SVM), and Decision Tree Classifier, along with insights derived from the analysis.

## Classification Overview
### Types of Classification Algorithms Used:
1. **k-Nearest Neighbors (k-NN)**
    - The k-NN algorithm predicts the classification of a data point by identifying the majority class among its k nearest neighbors in the feature space.

2. **Support Vector Machine (SVM)**
    - SVM is a supervised machine learning algorithm used for classification tasks. It finds the hyperplane that best separates classes in a high-dimensional space.

3. **Decision Tree Classifier**
    - Decision trees create models that predict the value of a target variable by learning simple decision rules inferred from the features.

## Data Analysis Insights
### Data Preprocessing
- **Gender Encoding:** The 'Gender' feature was transformed into binary values (0 and 1) to facilitate classification.
- **One-Hot Encoding:** The 'Property_Area' feature underwent one-hot encoding to convert categorical data into a format suitable for analysis.
- **Qualitative to Binary Conversion:** Columns like 'Married', 'Education', 'Self_Employed', and 'Loan_Status' were converted to binary (0/1) values for model training.

### Model Training and Evaluation
- **Train-Test Split:** The dataset was split into training and testing sets (80/20 split) to train and evaluate the models.
- **Model Performance:** k-NN, SVM, and Decision Tree Classifier models were trained and evaluated using confusion matrices to assess their performance.

### Feature Scaling
- **Standardization:** Features were standardized using StandardScaler to ensure all features contributed equally to model training.

### Decision Tree Visualization
- **Tree Visualization:** A Decision Tree Classifier with a maximum depth of 3 was created and visualized to understand the decision-making process.

## Launching project
### Running the Project
To run the project, you have a few options:

1. **Google Colab via Gist:**
   Use the Google Colab environment via the Gist website and include the learning data from the GitHub project resources. [Gist Link](https://gist.github.com/RobertNeat/5b2ad5a70382fb1fe342a44026eadf96)

2. **Local Environment:**
   - **DataSpell or PyCharm by JetBrains:** Download the repository branch and launch the project locally using DataSpell or PyCharm by JetBrains. Ensure dataset files are inside your project directory.
   - **Spyder IDE:** Alternatively, you can launch the project using the Spyder IDE. Remember to have the dataset files within your project directory to avoid any issues.

By following these steps, you can execute and explore the loan classification model in your preferred environment.

## Conclusions
The analysis provided valuable insights into the loan approval process. Features like marital status, education, and employment status seem to have a significant influence on loan approvals. The decision tree visualization highlighted key factors considered in the loan approval process, providing a clear view of the decision-making pathway.

This model can serve as a starting point for further refinements and improvements in predicting loan approvals, aiding financial institutions in making informed decisions.
