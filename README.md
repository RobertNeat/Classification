# Breast Cancer Classification Model

This repository contains code for a breast cancer classification model using the Scikit-learn library in Python.

### Classification Techniques Used:
The model employs three different classification techniques:

### 1. Decision Tree Classifier:
- **Description:** Decision tree algorithm to classify breast cancer instances.
- **Parameters:** Maximum depth set to 5.
- **Results:**
  - **Confusion Matrix:** 
    ```
    [[47  2]
     [ 1 64]]
    ```
  - **Sensitivity:** 0.959
  - **Precision:** 0.979
  - **Specificity:** 0.985
  - **Accuracy:** 0.974

### 2. k-Nearest Neighbors (kNN) Classifier:
- **Description:** kNN algorithm with 6 nearest neighbors.
- **Results:**
  - **Confusion Matrix:** 
    ```
    [[46  3]
     [ 0 65]]
    ```
  - **Sensitivity:** 0.939
  - **Precision:** 1.000
  - **Specificity:** 1.000
  - **Accuracy:** 0.974

### 3. Support Vector Machine (SVM) Classifier:
- **Description:** SVM algorithm using a polynomial kernel of degree 8.
- **Results:**
  - **Confusion Matrix:** 
    ```
    [[23 26]
     [ 0 65]]
    ```
  - **Sensitivity:** 0.469
  - **Precision:** 1.000
  - **Specificity:** 1.000
  - **Accuracy:** 0.772

### Conclusion:
- Decision Tree and kNN classifiers perform with high accuracy and precision.
- SVM, despite its precision, has lower sensitivity, impacting its overall performance on this dataset.
- Scaling the data notably improved the performance of kNN and SVM classifiers, while Decision Trees performed well without scaling.
- Further optimization and feature engineering could enhance the performance of the SVM classifier on this dataset.

For detailed code and implementation, refer to the provided Python file.

### Launching Project:
To run the project, you have two options:

1. **Google Colab via Gist:**
   - Use the Google Colab environment via the Gist website.
   - Access the project resources including the dataset from the following [gist_link_here](https://gist.github.com/RobertNeat/a07fec9d76fd48abf0d589ca08a0a64e).

2. **Local Deployment:**
   - Download the repository branch and launch the project locally.
   - Use IDEs like DataSpell or PyCharm by JetBrains, or Spyder IDE.
   - Ensure dataset files are within your project directory to prevent path-related issues.

Note: When launching locally, ensure the dataset files are present in your project directory.
