# Analysis of IMDb Movie Data: Exploring Success Factors
# Data Science Project Report: Movie Dataset Analysis

**#1. Problem Statement**

Aim:
The primary aim of this project is to develop a predictive model that classifies IMDb movie reviews as positive or negative and to extract insights into the sentiment and characteristics of movie reviews. This enables deeper understanding of audience feedback and helps answer data science questions related to sentiment trends, review quality, and content preferences.## 
#Key Data Science Questions:

What are the most common linguistic patterns and themes in positive and negative reviews?

How do sentiments vary across different genres and ratings?

Can sentiment analysis help identify fake or biased reviews?

What topics dominate positive or negative reviews?
**#2. Dataset**

**Source:**
IMDb Movie Reviews Dataset (Stanford Dataset).

**Feature Set:**
The dataset contains 50,000 movie reviews with the following key features:

Text reviews: The actual movie reviews written by users.

Sentiment labels: Binary labels indicating whether a review is positive or negative.

Metadata (optional): Includes features like movie title, genre, and IMDb rating for auxiliary analysis.
The dataset includes:
- *Features*: Director, actors, financial metrics (budget, gross earnings), IMDb score, and other movie details such as _name	num_critic_for_reviews	duration	director_facebook_likes	actor_3_facebook_likes	actor_2_name	actor_1_facebook_likes, genres, num_user_for_reviews, language, country, content_rating, title_year, actor_2_facebook_likes,imdb_score	aspect_ratio, movie_facebook_likes
  ![image](https://github.com/user-attachments/assets/29c49da5-62fa-41fd-8c3a-e013be3b9b46)
Purpose: This dataset helps analyze the text of movie reviews to determine their sentiment, using other attributes for auxiliary insights.
**Methodology:**
#**1. Data Preprocessing:**
The movie_metadata.csv dataset contains 28 columns and 5043 rows. Key features include:
these attributes has relationship among them and we'll explore the data by visualization with various libraries available in python to visualize data.

**Importing Libraries**
We need to import libraries that are required to process dataset, which will be using to clean data and which are using to visualize data
![image](https://github.com/user-attachments/assets/0f756e08-ac4a-4586-9d76-d3ed6e9f887a)
#****Loading the data into the data frame**  
n this section, we load the IMDb movie reviews dataset into a Pandas DataFrame. We explore the dataset by examining its structure, such as the number of rows and columns, and check for any missing or duplicate values. This initial exploration helps us understand the data better and prepare for preprocessing.
# Display first few rows
df.head()
![image](https://github.com/user-attachments/assets/b16b0001-b734-4efc-b57f-488d2c1ae24f)
# Display dataset dimensions
print(f"Dataset dimensions: {df.shape}")

# Display summary statistics
print("Dataset Summary:")
print(df.describe())
![image](https://github.com/user-attachments/assets/e06bf335-4e3d-4fdc-bd7f-0e87bee8d040)


*2. Data Cleaning and Preprocessing*
Here, we clean the dataset by addressing missing values and duplicates. We also perform text preprocessing steps such as:
- Removing special characters, numbers, and stopwords.
- Converting all text to lowercase.
- Tokenizing the reviews into words.
![image](https://github.com/user-attachments/assets/52eb324d-f990-4c37-9e1c-12bb9451686d)
![image](https://github.com/user-attachments/assets/ae946c9b-1f40-4eb0-8cb9-803a4367a9fe)
![image](https://github.com/user-attachments/assets/e309a598-9966-41c1-8b0f-3cf0e89a6331)
![image](https://github.com/user-attachments/assets/c730b214-f4fe-473c-9c2a-7d1680397e4f)
![image](https://github.com/user-attachments/assets/836cfcb5-1450-4120-8225-aefa62cdc100)
*** **Exploratory Data Analysis (EDA)*****
In this section, we perform EDA to visualize and summarize the dataset. This includes:
- Analyzing the distribution of positive and negative reviews.
- Using heat map, bar charts and histograms to understand key trends in the data.

![image](https://github.com/user-attachments/assets/6d228a03-143b-4e53-b2e1-52935450aa82)

df.columns
![image](https://github.com/user-attachments/assets/44a8a3ba-6587-4397-a9e5-c37f31d3fc80)
![image](https://github.com/user-attachments/assets/57ad9d67-7493-4d8b-8e0a-89c220bafce6)

![image](https://github.com/user-attachments/assets/229cfeb8-140e-45f0-8c45-f9e45188ebb9)

**Line Chart**
![image](https://github.com/user-attachments/assets/23efd9b3-e83a-4782-bddd-a2241795689d)
# 2.Boxplot: Distribution of Movie Budgets
box_data = df[['budget']].dropna()

plt.figure(figsize=(8, 6))
sns.boxplot(data=box_data, x='budget', color='skyblue')
plt.title('Distribution of Movie Budgets', fontsize=14)
plt.xlabel('Budget', fontsize=12)
plt.show()


3. Sentiment Analysis:
    - Supervised learning: train a classifier on labeled data.
    - Models: Logistic Regression, Support Vector Machine (SVM), Random Forest, Convolutional Neural Network (CNN).
    - Evaluate performance using accuracy, precision, recall, F1-score, and ROC-AUC.
4. Topic Modeling:
    - Latent Dirichlet Allocation (LDA) for topic extraction.
    - Identify underlying themes and sentiments in reviews.
5. Visualization:
    - Sentiment distribution across genres and ratings.
    - Word cloud representation of frequent words in positive/negative reviews.


