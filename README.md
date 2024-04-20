
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
  - **Outputs:** Clusters users into groups based on their movie watching patterns and preferences.
- **Linear Regression Model:**
  - **Inputs:** User ID
  - **Outputs:** Uses Linear Regression to predict Movies using User ID.

  KNN Model :
      The initial phase of our project involved merging the ratings and movies datasets. During this process, it became apparent that a considerable number of movies had received zero ratings compared to other ratings. To address this imbalance, we applied a Log Transform to the count of each rating.The recommendation system was engineered to predict user preferences across a diverse array of movies. To achieve this, the refined dataset underwent transformation into a user-movie matrix. In this matrix:
    
    \begin{itemize}
        \item Rows corresponded to individual users.
        \item Columns represented movie titles.
        \item Entries in the matrix indicated the average rating given by a user to a specific movie.

## Contributing
We welcome contributions to improve the recommendation algorithms or any other aspects of the system. Please follow the standard fork-branch-pull request workflow.

## Authors
- **Arman Gupta**
- **Soumen Kumar**
- **Sangini Garg**
- **Anuj Chicholikar**
- **Anushka Dadhich**
- **Yogita Mundankar**
- **Abhay Kashyap**

---
