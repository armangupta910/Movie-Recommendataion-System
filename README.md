
# Movie Recommendation System

## Overview
This repository contains a movie recommendation system that employs multiple machine learning models to provide personalized movie recommendations based on different sets of features. The models used include K-Nearest Neighbors (KNN), Artificial Neural Networks (ANN), Support Vector Machines (SVM), and Gaussian Mixture Models (GMM). The recommendation system works based on the Movies Lens dataset containing 100k Ratings.

## Models and Features
The recommendation system utilizes the following models and features:

- **K-Nearest Neighbors (KNN):**
  - **Inputs:** User ID, Name of the Movie
  - **Outputs:** Provides recommendations by finding similar users based on movie preferences.
    

- **Artificial Neural Network (ANN):**
  - **Inputs:** Genres, Ratings, User Feedback
  - **Outputs:** Predicts user preferences based on non-linear relationships in movie features and user feedback.

- **Support Vector Machines (SVM):**
  - **Inputs:** User ID, User Feedback
  - **Outputs:** Classifies users into different preference groups to suggest movies that match their tastes.

- **Gaussian Mixture Model (GMM):**
  - **Inputs:** User ID
  - **Outputs:** Clusters users into groups based on their movie-watching patterns and preferences.
- **Linear Regression Model:**
  - **Inputs:** User ID
  - **Outputs:** Uses Linear Regression to predict Movies using User ID.

  **KNN Model** :
      The initial phase of our project involved merging the ratings and movie datasets. During this process, it became apparent that many movies had received zero ratings compared to others. To address this imbalance, we applied a Log Transform to the count of each rating. The recommendation system was engineered to predict user preferences across various movies. To achieve this, the refined dataset underwent a transformation into a user-movie matrix. In this matrix:
    
    Rows corresponded to individual users.
    Columns represented movie titles.
    Entries in the matrix indicated the average rating a user gives to a specific movie.

  Metric: Cosine similarity, chosen for its capacity to compare users regardless of the number of ratings they have submitted. This metric focuses on the angle between rating vectors rather than their magnitude, rendering it ideal for comparing similarity in sparse data.
  Algorithm: Brute force, selected for its comprehensive evaluation of distances between all pairs of points in the dataset. Despite its computational intensity, this method ensures that no potential connections are overlooked, a critical consideration in recommendation 
  contexts where accuracy is paramount.


  **SVR Model**: SVR is a powerful machine-learning technique that can be used for regression tasks, such as predicting movie ratings. In this implementation, an SVR model is initialized with specific parameters, including the kernel function (rbf), regularization 
  parameter (C=1.0), and epsilon value (epsilon=0.2). The model is trained using the flattened training data, and predictions are made on the flattened test data.
  The performance of the SVR model is evaluated using three widely used metrics:

  Mean Squared Error (MSE): Measures the average squared difference between the predicted and actual ratings.

  Mean Absolute Error (MAE): Measures the average absolute difference between the predicted and actual ratings.

  R^2 Score: Indicates the proportion of the variance in the target variable that is explained by the model.

  **GMM Model**:
  Made predictions on the test set using the best model obtained from grid search.
  - Calculated the Root Mean Squared Error (RMSE) between predicted and actual ratings to evaluate model performance.
  - Confidence Score Calculation: The confidence scores represent the maximum probability assigned to each prediction by the model. The maximum probability across all possible rating classes is determined for each prediction, indicating the model's confidence in its 
  prediction.

  **ANN Model**:
  Data Preparation:
  Encoded categorical variable 'genres' using LabelEncoder.
  Scaled numerical feature 'rating' using MinMaxScaler.
  Encoding Categorical Variables:
  Utilized LabelEncoder to transform categorical 'genres' into numerical labels.
  Scaling Numerical Features:
  Applied MinMaxScaler to scale the 'rating' feature to a range between 0 and 1.
         
  Model Building:
      Constructed an ANN model for movie recommendation.
      Defined input layers for genre and rating.
      Utilized Embedding layer for genre input and Concatenate layer to merge inputs.
      Comprised Dense layers for learning complex patterns.
      Compiled the model using mean squared error loss and Adam optimizer.

  ## Contributing
  We welcome contributions to improve the recommendation algorithms or any other aspects of the system. Please follow the standard fork-branch-pull request workflow.

## Link to Report
 https://drive.google.com/file/d/11RGwQ5BXFuh7kwRW2_cEo1Mito5s-59K/view

  ## Link to video presentation
  https://youtu.be/D73zJuKNLNc

  ## Link to presentation
  https://drive.google.com/file/d/1Zg5G1_3zQ6LxQSRjnXDOFdgKOE4mgj7n/view

  ## Authors
  - **Arman Gupta**
  - **Soumen Kumar**
  - **Sangini Garg**
  - **Anuj Chicholikar**
  - **Anushka Dadhich**
  - **Yogita Mundankar**
  - **Abhay Kashyap**

---
