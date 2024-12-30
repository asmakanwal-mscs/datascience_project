# Analysis of IMDb Movie Data: Exploring Success Factors
# Data Science Project Report: Movie Dataset Analysis

**#1. Problem Statement**  
**Aim:**  
The primary aim of this project is to develop a predictive model that classifies IMDb movie reviews as positive or negative and to extract insights into the sentiment and characteristics of movie reviews. This enables deeper understanding of audience feedback and helps answer data science questions related to sentiment trends, review quality, and content preferences.## 
**#Key Data Science Questions:**  
1. How can sentiment analysis of IMDb reviews influence marketing strategies for upcoming films? - This question focuses on understanding how audience sentiment can shape promotional efforts and targeting strategies.
2. What features of movie reviews are most strongly correlated with positive or negative sentiments? - Investigating which aspects of reviews (e.g., acting, direction, plot ) most impact audience sentiment can help filmmakers focus on strengths.
3. How do sentiment trends change over time for a specific movie after its release? - This question examines how public perception evolves, which can inform marketing and distribution strategies.
4. What is the sentiment distribution of reviews across different genres of movies? - Analyzing whether certain genres tend to receive more positive or negative feedback can guide future film production decisions. 5. How do sentiments expressed in IMDb reviews compare to box office performance? - This question seeks to establish a correlation between sentiment analysis results and actual financial outcomes, providing insights into predictive modeling.
**#2. Dataset**  
The movie_metadata.csv dataset contains 28 columns and 5043 rows.      
**Source:**    
IMDb Movie Reviews Dataset (Stanford Dataset).  
https://www.kaggle.com/datasets/carolzhangdc/imdb-5000-movie-dataset?resource=download
**Feature Set:**    
The dataset contains 50,000 movie reviews with the following key features:
'color', 'director_name', 'num_critic_for_reviews', 'duration', 'director_facebook_likes', 'actor_3_facebook_likes', 'actor_2_name',
'actor_1_facebook_likes', 'gross', 'genres', 'actor_1_name', 'movie_title', 'num_voted_users', 'cast_total_facebook_likes',
       'actor_3_name', 'facenumber_in_poster', 'plot_keywords','movie_imdb_link', 'num_user_for_reviews', 'language', 'country',
       'content_rating', 'budget', 'title_year', 'actor_2_facebook_likes','imdb_score', 'aspect_ratio', 'movie_facebook_likes'],
      dtype='object')
**Text reviews:** The actual movie reviews written by users.    
**Sentiment labels:** Binary labels indicating whether a review is positive or negative.    
Metadata (optional): Includes features like movie title, genre, and IMDb rating for auxiliary analysis.
The dataset includes:    
- ***Features*:** Director, actors, financial metrics (budget, gross earnings), IMDb score, and other movie details such as _name	num_critic_for_reviews	duration	director_facebook_likes	actor_3_facebook_likes	actor_2_name	actor_1_facebook_likes, genres, num_user_for_reviews, language, country, content_rating, title_year, actor_2_facebook_likes,imdb_score	aspect_ratio, movie_facebook_likes  
  ![image](https://github.com/user-attachments/assets/29c49da5-62fa-41fd-8c3a-e013be3b9b46)
Purpose: This dataset helps analyze the text of movie reviews to determine their sentiment, using other attributes for auxiliary insights.  
#**Methodology:**     
#**1. Data Preprocessing:**  
The movie_metadata.csv dataset contains 28 columns and 5043 rows. Key features include:   
these attributes has relationship among them and we'll explore the data by visualization with various libraries available in python to visualize data.  
**Importing Libraries**  
We need to import libraries that are required to process dataset, which will be using to clean data and which are using to visualize data
![image](https://github.com/user-attachments/assets/0f756e08-ac4a-4586-9d76-d3ed6e9f887a)  
****Loading the data into the data frame**     
n this section, we load the IMDb movie reviews dataset into a Pandas DataFrame. We explore the dataset by examining its structure, such as the number of rows and columns, and check for any missing or duplicate values. This initial exploration helps us understand the data better and prepare for preprocessing.  
**Display first few rows  **
df.head()  
![image](https://github.com/user-attachments/assets/b16b0001-b734-4efc-b57f-488d2c1ae24f)    
**Display dataset dimensions**  
print(f"Dataset dimensions: {df.shape}")
(5043, 28)
**Display summary statistics**  
print("Dataset Summary:")  
print(df.describe())  
![image](https://github.com/user-attachments/assets/e06bf335-4e3d-4fdc-bd7f-0e87bee8d040)  
***Data Cleaning and Preprocessing**  
Here, we clean the dataset by addressing missing values and duplicates. We also perform text preprocessing steps such as:  
- Removing special characters, numbers, and stopwords.  
- Converting all text to lowercase.  
- Tokenizing the reviews into words.  
![image](https://github.com/user-attachments/assets/52eb324d-f990-4c37-9e1c-12bb9451686d)  
![image](https://github.com/user-attachments/assets/ae946c9b-1f40-4eb0-8cb9-803a4367a9fe)  
![image](https://github.com/user-attachments/assets/e309a598-9966-41c1-8b0f-3cf0e89a6331)  
![image](https://github.com/user-attachments/assets/c730b214-f4fe-473c-9c2a-7d1680397e4f)  
![image](https://github.com/user-attachments/assets/836cfcb5-1450-4120-8225-aefa62cdc100)  
****Exploratory Data Analysis (EDA)****  
In this section, we perform EDA to visualize and summarize the dataset. This includes:  
- Analyzing the distribution of positive and negative reviews.
- Using heat map, bar charts and histograms to understand key trend  s in the data.  

![image](https://github.com/user-attachments/assets/6d228a03-143b-4e53-b2e1-52935450aa82)  
  
**df.columns**  
![image](https://github.com/user-attachments/assets/44a8a3ba-6587-4397-a9e5-c37f31d3fc80)  
![image](https://github.com/user-attachments/assets/57ad9d67-7493-4d8b-8e0a-89c220bafce6)   
![image](https://github.com/user-attachments/assets/229cfeb8-140e-45f0-8c45-f9e45188ebb9)  

# **1. Line Chart**  
The line chart shows the average IMDb score over the years:  
X-axis (Year): Movie release year. Y-axis (IMDb Score): Average IMDb score. Line: Tracks the trend over time. It visualizes the change in average IMDb scores across the years.  
![image](https://github.com/user-attachments/assets/23efd9b3-e83a-4782-bddd-a2241795689d)  
#** 2.Boxplot: Distribution of Movie Budgets**  
The boxplot shows the distribution of movie budgets:  
Box: Represents the middle 50% of budgets. Median Line: Shows the median budget. Whiskers: Indicate the range, excluding outliers. Outliers: Show exceptionally high or low budgets. It visualizes the spread and central tendency of budgets.  
box_data = df[['budget']].dropna()
plt.figure(figsize=(8, 6))
sns.boxplot(data=box_data, x='budget', color='skyblue')
plt.title('Distribution of Movie Budgets', fontsize=14)
plt.xlabel('Budget', fontsize=12)
plt.show()
![image](https://github.com/user-attachments/assets/4550b4e3-038d-4f88-a370-6a152dc46327)
# 3.Bar Graph: Count of Movies Per Genre#
The bar graph shows the count of movies per genre. 
X-axis (Genre): Represents the movie genres. Y-axis (Count): Shows the number of movies in each genre. 
Bar Height: Taller bars indicate more movies in that genre. 
Rotation: Genre labels are rotated for better readability. the bar graph visualizes the distribution of movies across different genres.
df['genres'] = df['genres'].fillna('Unknown')
genre_counts = df['genres'].str.split('|').explode().value_counts()
plt.figure(figsize=(12, 6))
sns.barplot(x=genre_counts.index, y=genre_counts.values, palette='viridis')
plt.title('Count of Movies Per Genre', fontsize=14)
plt.xlabel('Genre', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=45)
plt.show()
![image](https://github.com/user-attachments/assets/ae1e0556-8f09-41bd-9f67-b2709082a19f)

# 4.Heatmap: Correlation Between Numeric Variables#
The heatmap shows correlations between numeric variables:
Color Intensity: Indicates the strength of the correlation (red for positive, blue for negative). 
Values: Correlation coefficients range from -1 to 1. 
Annotations: Display the exact correlation values. 
It highlights relationships between variables in the dataset.
![image](https://github.com/user-attachments/assets/36397b81-58d5-4721-9676-6e3736606b00)

# Select only numeric columns
numeric_df = df.select_dtypes(include=['float64', 'int64'])

# Compute the correlation matrix
correlation_matrix = numeric_df.corr()

# Plot the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', cbar=True)
plt.title('Correlation Heatmap', fontsize=14)
plt.show()
![image](https://github.com/user-attachments/assets/090c2c2e-e2c8-4388-9187-3376510b8070)
X-axis (IMDb Score): Represents the IMDb score range. 
Y-axis (Frequency): Shows how many movies fall within each IMDb score range. 
KDE (Kernel Density Estimate): The smooth curve shows the overall distribution of scores. 
Bins: The number of bins (20) helps visualize the distribution in more detail. 
In summary, the histogram helps understand the spread and concentration of IMDb scores across movies, highlighting common score ranges and any skewness.
# 5.Histogram: Distribution of IMDb Scores
plt.figure(figsize=(8, 6))
sns.histplot(df['imdb_score'], kde=True, bins=20, color='purple')
plt.title('Distribution of IMDb Scores', fontsize=14)
plt.xlabel('IMDb Score', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.show()
![image](https://github.com/user-attachments/assets/0febaa29-35a8-433d-a446-0033cab3cbc2)
The scatter plot shows the relationship between gross earnings and budget. Key Insights: X-axis (Budget): Displays the movie budget. Y-axis (Gross Earnings): Shows the gross earnings. Point Distribution: Each point represents a movie, and the spread indicates the correlation between budget and earnings. Alpha Transparency: Makes overlapping points visible, helping to understand data density, this scatter plot visually shows how budget correlates with earnings across movies.
# 6.Scatter Plot: Gross Earnings vs Budget  
scatter_data = df[['budget', 'gross']].dropna()

plt.figure(figsize=(10, 6))
sns.scatterplot(data=scatter_data, x='budget', y='gross', alpha=0.6, color='green')
plt.title('Gross Earnings vs Budget', fontsize=14)
plt.xlabel('Budget', fontsize=12)
plt.ylabel('Gross Earnings', fontsize=12)
plt.grid(True)
plt.show()
![image](https://github.com/user-attachments/assets/a8155d6b-4367-4797-9942-1db513e4420f)
The treemap shows the proportion of movies by genre:

Larger Rectangles: Represent genres with more movies. Smaller Rectangles: Represent genres with fewer movies. Color Palette: Differentiates genres visually. Key Insight: Easily identify the most and least common genres in the dataset. If "Unknown" is a large block, it indicates missing or incomplete genre data.
# 7.Treemap: Proportion of Movies by Genre    
!pip install squarify
import squarify

# Prepare data for treemap  
genre_counts = df['genres'].str.split('|').explode().value_counts()

plt.figure(figsize=(12, 6))
squarify.plot(sizes=genre_counts.values, label=genre_counts.index, alpha=0.8, color=sns.color_palette('pastel'))
plt.axis('off')
plt.title('Proportion of Movies by Genre', fontsize=14)
plt.show()
![image](https://github.com/user-attachments/assets/91ed5c72-1aff-4cd7-8caf-5b56cb24bcc8)
The pair plot shows the relationships between IMDb score, budget, and gross earnings:

Diagonal: Displays the distribution of each variable. Off-diagonal: Shows pairwise relationships: IMDb score vs. Budget IMDb score vs. Gross Budget vs. Gross Alpha Transparency: Helps see overlapping points. Key insights: Identify correlations, trends, and outliers between these key variables.
# 8.Pair Plot: Relationship Between Key Variables
sns.pairplot(df, vars=['imdb_score', 'budget', 'gross'], diag_kind='kde', corner=True, plot_kws={'alpha': 0.7})
plt.suptitle('Pairwise Relationships Between Key Variables', y=1.02, fontsize=14)
plt.show()
![image](https://github.com/user-attachments/assets/a204974d-b34b-49a8-b714-024e89e1630b)
The word cloud displays the most common plot keywords:
Larger Words: Represent more frequent keywords. Smaller Words: Represent less frequent keywords. It visually highlights recurring themes or concepts in movie plots.
# 9.Word Cloud: Common Words in Plot Keywords
!pip install wordcloud
from wordcloud import WordCloud

# Combine all keywords into a single string
text = ' '.join(df['plot_keywords'].dropna())

# Generate word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

# Plot the word cloud
plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Plot Keywords', fontsize=14)
plt.show()
![image](https://github.com/user-attachments/assets/c13cebc1-c014-4bc1-b069-ad9377ba112f)
The pie chart shows the proportion of movies by content rating:
Larger Slices: Represent more movies in that rating. Percentage Labels: Indicate the share of each content rating. It visually highlights the distribution of movies across content ratings.
# 10.Pie Chart: Proportion of Movies by Content Rating
content_rating_counts = df['content_rating'].value_counts()

plt.figure(figsize=(8, 8))
plt.pie(content_rating_counts, labels=content_rating_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))
plt.title('Proportion of Movies by Content Rating', fontsize=14)
plt.axis('equal')
plt.show()
![image](https://github.com/user-attachments/assets/819ce89c-0a34-4a3d-bdd2-87c773034148)
The bubble chart shows budget vs. gross earnings:

X-axis: Budget Y-axis: Gross earnings Bubble Size: Proportional to IMDb score (larger bubbles = higher IMDb scores). It visualizes the relationship between budget, earnings, and IMDb scores.
# 11.Bubble Chart: Budget vs Gross Earnings with IMDb Scores
bubble_data = df[['budget', 'gross', 'imdb_score']].dropna()

plt.figure(figsize=(10, 6))
plt.scatter(bubble_data['budget'], bubble_data['gross'],
            s=bubble_data['imdb_score']*20, alpha=0.6, color='teal')
plt.title('Budget vs Gross Earnings with IMDb Scores', fontsize=14)
plt.xlabel('Budget', fontsize=12)
plt.ylabel('Gross Earnings', fontsize=12)
plt.grid(True)
plt.show()
![image](https://github.com/user-attachments/assets/a5bc3436-1be0-4c4f-8aa3-8c05cfd45246)
The violin plot shows the distribution of IMDb scores by content rating:
Width: Indicates the density of IMDb scores at different levels. Median Line: Shows the median IMDb score for each content rating. Shape: Reveals the spread and distribution of scores across content ratings.
# 12.Violin Plot: IMDb Scores by Content Rating
plt.figure(figsize=(12, 6))
sns.violinplot(data=df, x='content_rating', y='imdb_score', palette='muted')
plt.title('Distribution of IMDb Scores by Content Rating', fontsize=14)
plt.xlabel('Content Rating', fontsize=12)
plt.ylabel('IMDb Score', fontsize=12)
plt.xticks(rotation=45)
plt.show()
![image](https://github.com/user-attachments/assets/63b98c73-19f4-44f8-a042-150feb0897cb)
The stacked bar chart shows movie count by year and content rating:
X-axis: Year of release Y-axis: Number of movies Stacked Bars: Represent movie counts split by content rating. It visualizes movie trends over the years by content rating.
# 13.Stacked Bar Chart: Movie Count by Year and Content Rating
stacked_data = df.groupby(['title_year', 'content_rating']).size().unstack(fill_value=0)

stacked_data.plot(kind='bar', stacked=True, figsize=(14, 8), colormap='cool')
plt.title('Movie Count by Year and Content Rating', fontsize=14)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Number of Movies', fontsize=12)
plt.legend(title='Content Rating', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
![image](https://github.com/user-attachments/assets/3fd0a9b6-b5f7-415f-aeaf-11023703a369)
The radar chart compares performance metrics for the top 10 movies:
Metrics: IMDb score, budget, gross earnings, and user reviews. Normalized Data: All metrics are scaled for easier comparison. Shape: Represents each movie’s performance across these metrics. It provides a visual comparison of top movies' key performance indicators.
# 14.Radar Chart: Performance Metrics for Top Movies
!pip install pandas.plotting
from pandas.plotting import radviz

# Select top-rated movies
top_movies = df.nlargest(10, 'imdb_score')[['imdb_score', 'budget', 'gross', 'num_user_for_reviews']].dropna()

# Normalize the data for radar chart
normalized_data = (top_movies - top_movies.min()) / (top_movies.max() - top_movies.min())

plt.figure(figsize=(8, 8))
radviz(normalized_data, class_column='imdb_score')
plt.title('Radar Chart: Performance Metrics for Top Movies', fontsize=14)
plt.show()

----
df['genres'].value_counts()
Comedy|Drama|Romance                              105
Comedy|Romance                                     85
Comedy|Drama                                       82
Drama                                              79
Comedy                                             77
                                                 ... 
Action|Adventure|Drama|Family                       1
Action|Adventure|Comedy|Crime|Thriller              1
Adventure|Animation|Family|Fantasy|Musical|War      1
Action|Adventure|Comedy|Drama|Thriller              1
Comedy|Documentary|Drama                            1
Name: genres, Length: 511, dtype: int64
---Content Rating Titles
df.content_rating.value_counts().nsmallest(50).plot(kind='bar',figsize=(10,5),color="red")
plt.title('content ratingTitle with the fewest numberz')
plt.ylabel('content_rating')
plt.xlabel('actor_2_name')
plt.show()
![image](https://github.com/user-attachments/assets/28b9cc09-b6c9-47e1-9b30-bbb3ac8d2087)
--quantile---
q1=df.quantile(0.25)
q3=df.quantile(0.75)
IQR=q3-q1
print(IQR)
![image](https://github.com/user-attachments/assets/0c61b677-d35b-4ec3-a296-9537808fac7e)
df=df[~((df<(q1-1.5*IQR)) |(df>(q3+1.5*IQR))).any(axis=1)]
df.shape
C:\Users\Navya12\AppData\Local\Temp\ipykernel_16240\3820765173.py:1: FutureWarning: Automatic reindexing on DataFrame vs Series comparisons is deprecated and will raise ValueError in a future version. Do `left, right = left.align(right, axis=1, copy=False)` before e.g. `left == right`
  df=df[~((df<(q1-1.5*IQR)) |(df>(q3+1.5*IQR))).any(axis=1)]
(2043, 28)
**Removing Outliers**
#Finding 1st Quartile, 3rd Quartile and Interquartile Range


Q1 = df.quantile(0.25, numeric_only = True)
Q3 = df.quantile(0.75, numeric_only = True)
IQR = Q3 - Q1
print('Q1(quantile - 0.25) = ', Q1)
print('----------------------------------------------')
print('Q3(quantile - 0.75) = ', Q3)
print('----------------------------------------------')
print('IQR(Q3 - Q1) = ', IQR)
![image](https://github.com/user-attachments/assets/24cab09b-d48e-4dd7-b3a6-d812d9beeb84)

![image](https://github.com/user-attachments/assets/45dcfcb9-ae90-4860-8f2e-0504f13dcef5)
![image](https://github.com/user-attachments/assets/053c3b3e-6ab7-4ff6-9845-46cb338b849d)
![image](https://github.com/user-attachments/assets/abc9de31-3016-45bb-9948-bcda7b3d2659)

# Insights from Applied ML Models  

**1. Regression Model (IMDb Score Prediction)**  
 Key Features: Budget, gross earnings, user reviews, and duration. Performance: RMSE quantifies prediction error. Example: RMSE = 0.85 indicates average deviation of 0.85 IMDb points. Feature Importance: User reviews and gross earnings likely contribute the most to predictions. Limitations: Subjective factors like audience sentiment aren't captured, and missing/skewed data affects accuracy.  

**2. Classification Model (Content Rating Prediction) **  
 Key Features: Budget, gross earnings, user reviews. Performance: Example accuracy = 82%. Precision and recall reveal strengths/weaknesses for each rating. Feature Importance: Budget and user reviews help distinguish ratings. Limitations: Imbalanced data and subjective criteria (e.g., violence) impact predictions.
General Takeaways Random Forest models provide interpretable predictions but are sensitive to data quality. Enhancements: Include richer features (e.g., genres, director), and optimize hyperparameters for better accuracy.


[ ]
![image](https://github.com/user-attachments/assets/c3a80b5e-d2ef-43d7-bc93-3d3a27b7d3e6)
![image](https://github.com/user-attachments/assets/92aef67c-a5e1-4705-8515-ae3247f1808a)
![image](https://github.com/user-attachments/assets/afcfbcc7-dc03-4d82-a9d7-b3174e09ed0c)
![image](https://github.com/user-attachments/assets/c0ff34a8-2201-449f-96cb-bff43a954b7d)
**Machine Learning Integration Steps:** Regression for IMDb score prediction. Classification for content rating prediction. Evaluate models using RMSE, accuracy, and classification reports. Visualize results for better insight.

# ***Key Insights*  
*High Positive Correlations:*  
# Based on **Heat map**  
![image](https://github.com/user-attachments/assets/0ad06696-9c6e-4312-8b5c-5fc019058561)  

num_voted_users and gross (Correlation: 0.64): This suggests that **movies with more user votes** tend to **have higher gross earnings**.  
imdb_score and num_critic_for_reviews (Correlation: 0.48) and imdb_score and num_user_for_reviews (Correlation: 0.45): Higher IMDb scores are associated with more reviews from both  
 critics and users.  
**Moderate Positive Correlations:**    
movie_facebook_likes and num_voted_users (Correlation: 0.47): Movies with more Facebook likes tend to have more user votes.    
budget and gross (Correlation: 0.45): Higher budget movies tend to have higher gross earnings.    
**Insights Related to Data Science Questions:**    
**Influencing Marketing Strategies:**    
Understanding the positive correlation between movie Facebook likes and gross earnings can help in designing marketing strategies that focus on social media engagement.    
**Features Correlated with Sentiments:**    
The correlation between IMDb scores and the number of critic/user reviews suggests that these features might be significant in predicting the sentiment of reviews.    
**Sentiment Distribution Across Genres:**    
A deeper analysis of correlations within specific genres (not shown in this heatmap) would be needed to answer this question.    
**Comparison to Box Office Performance:**     
The positive correlation between num_voted_users and gross indicates that user engagement, reflected in the number of votes, might be a good predictor of box office performance**.    
# **Histogram**    
![image](https://github.com/user-attachments/assets/2d087db0-f04e-42da-bff5-d7e4ab3d1ff0)    
**Key insights:**    
Distribution: The histogram shows that **most IMDb scores are centered around 6 to 7**. This indicates that a majority of the movies in your dataset receive **average ratings**.    
Frequency Peaks: There are **fewer movies with extremely high** (above 9) or **extremely low** (below 4) scores. This suggests that extreme ratings are **less common**.     
KDE Line:(Kernel Density Estimate, the KDE line smooths out the data to give a clearer picture of the overall distribution. The smooth KDE line overlaying the histogram gives a clearer  
 picture of the distribution's shape, confirming that the scores follow a roughly normal distribution, with a peak around the average scores.)    
**Insights Related to Your Data Science Questions:**     
**Influencing Marketing Strategies:**    
Knowing that most movies have average ratings, marketing strategies could focus on **emphasizing unique selling points that can boost a movie's rating** above the average threshold.    
**Features Correlated with Sentiments:**    
The central tendency around 6 to 7 suggests that **features leading to above-average ratings should be explored further to understand what contributes to higher sentiment.**    
**Comparison to Box Office Performance:**  
Comparing this distribution with box office performance data can help determine if higher-rated movies tend to perform better financially.  
# **Scatter Plot**  
![image](https://github.com/user-attachments/assets/ea6cd920-667c-45bc-ba2f-d268ae4c30bd)  
The scatter plot visualizes the relationship between the budget and gross earnings of movies in your IMDb dataset. Each point represents a movie, with the x-axis showing the budget and the y-axis showing the gross earnings.  
**Key Observations:**  
**Clustering at Low Values:**  
Most of the data points are clustered near the lower end of both axes, indicating that many movies have relatively low budgets and gross earnings.  
**Outliers:  **
There are a few outliers with significantly higher budgets and gross earnings. These points highlight blockbuster movies with substantial financial investment and returns.  
**Positive Correlation:**  
There appears to be a **positive correlation between budget and gross earnings,** suggesting that **higher-budget movies tend to have higher gross earnings.**  
**Insights Related to Your Data Science Questions:**  
**Influencing Marketing Strategies:**  
Understanding the positive correlation between budget and gross earnings can **help marketing teams allocate resources more effectively, focusing on promoting high-budget movies** that are likely to yield higher returns.  
**Features Correlated with Sentiments:**  
While this plot specifically shows budget and gross earnings, it implies that financial investment in a movie can be correlated with its success, which may also impact audience sentiment positively.
**Comparison to Box Office Performance:**
The positive correlation between budget and gross earnings supports the idea that financial investment in movies is a strong predictor of box office performance, which can be further analyzed alongside sentiment data.
![image](https://github.com/user-attachments/assets/e7c8074e-99c7-4961-a536-44d93c656363)
The treemap chart titled "Proportion of Movies by Genre" visually represents the distribution of different movie genres by their proportion. Each genre is represented by a rectangle, with the size of the rectangle indicating the proportion of movies in that genre. The genres included are:

Thriller

Comedy

Drama

Horror

Fantasy

Sci-Fi

Adventure

Action

Animation

Biography

Family

Crime

Romance

Mystery

Sport

History

Music

War

Western

Film-Noir

Documentary

Musical

Reality-TV

**Key Observations:  
Genre Distribution:
** 
The chart shows that some genres, like **Drama and Comedy**, have larger rectangles, indicating a **higher proportion **of movies. In contrast, genres like Western and** Film-Noir have smaller rectangles, showing they are less common.**

**Proportion Insight:**
The treemap provides a visual understanding of how diverse the movie industry is in terms of genres and helps identify which genres dominate.
**Insights Related to Your Data Science Questions:  
Influencing Marketing Strategies:
**  
Understanding the proportion of movies by genre can help in tailoring marketing strategies. For example, if Drama and Comedy are more common, marketing efforts can focus on **differentiating these movies from others in the same genre.**  
Features Correlated with Sentiments:
Analyzing which genres have higher proportions can help identify features that contribute to positive or negative sentiments in those specific genres.
Trends Over Time:
This chart doesn’t show time-based trends, but understanding the current genre proportions can be a starting point for analyzing how these proportions change over time.
Sentiment Distribution Across Genres:
The treemap can be used to compare sentiment analysis results across different genres. Identifying which genres tend to receive more positive or negative feedback can guide future film production decisions.

# Pairwise Relationship between Key Variables  
![image](https://github.com/user-attachments/assets/f3552ae7-b722-49b7-9520-4754c7d121c1)   
The pair plot visualizes the relationships between three key variables in your IMDb movie reviews dataset:** imdb_score, budget, and gross.** The plot includes scatter plots for each pair of variables and histograms to show the distribution of each individual variable.  
**Key Insights:**  
Distributions:  
**IMDb Score:** The histogram shows that most IMDb scores are c**entered around 6 to 7,** which aligns with the previous histogram we discussed.  
**Budget:** The histogram indicates that a **majority of movies have relatively low budgets**, with a **few high-budget outliers.** 
**Gross:** Similar to the budget, the histogram shows that **most movies have lower gross earnings,** with a **few movies achieving significantly higher earnings**.
  
**Scatter Plots:**
**Budget vs. Gross**: There is a **positive correlation** between budget and gross, suggesting that higher-budget movies tend to earn more.  
IMDb Score vs. Budget: The scatter plot shows a weak positive correlation between IMDb score and budget, indicating that higher-budget movies might receive slightly higher ratings.  
IMDb Score vs. Gross: There is a weak positive correlation between IMDb score and gross, suggesting that higher-rated movies tend to earn more, though the relationship is not very    strong.
Insights Related to Your Data Science Questions:  
Influencing Marketing Strategies:  
Understanding the correlation between budget and gross earnings can help in designing marketing strategies that **focus on high-budget movies with potential for high returns**.
**Features Correlated with Sentiments:  **
The pair plot indicates that budget and gross earnings have some relationship with IMDb scores, which could be useful features for predicting the sentiment of reviews.
Comparison to Box Office Performance:

The positive correlations between IMDb scores, budget, and gross earnings support the idea that higher-rated and higher-budget movies tend to perform better financially, which can be useful for predictive modeling.

This pair plot provides a comprehensive visualization of how these key variables interact, offering valuable insights for your analysis. If you have more visualizations to share, please go ahead!  

# Conclusion
Analyzing the IMDb movie reviews dataset through various visualizations has provided significant insights into the sentiment and characteristics of movie reviews. Here are the key takeaways from our analysis:

# Influencing Marketing Strategies:

The positive correlation between movie budgets and gross earnings suggests that higher-budget movies tend to perform better financially. Marketing strategies can focus on promoting these high-budget movies to maximize returns. Additionally, leveraging social media engagement, as indicated by the correlation between Facebook likes and gross earnings, can be a powerful tool for marketing upcoming films.

# Features Correlated with Sentiments:

Features such as the number of user and critic reviews are positively correlated with IMDb scores, indicating that movies with more reviews tend to receive higher ratings. This suggests that aspects like acting, direction, and plot, which likely influence reviews, are critical to audience sentiment. Filmmakers can focus on these strengths to enhance the overall reception of their movies.
  
# Trends Over Time:  
  
While the current visualizations do not directly show trends over time, understanding how the distribution of IMDb scores evolves post-release can provide valuable insights into how public perception changes. Analyzing variables like the number of reviews and gross earnings over time can help in adjusting marketing and distribution strategies to maintain audience interest.  

# Sentiment Distribution Across Genres:  

The genre distribution treemap revealed that certain genres like Drama and Comedy dominate the movie industry. Analyzing sentiment distribution across these genres can guide future film production decisions by identifying which genres tend to receive more positive or negative feedback, helping filmmakers focus on genres that are more likely to be well-received by audiences.  
  
Comparison to Box Office Performance:  
  
The positive correlations between IMDb scores, budget, and gross earnings support the idea that higher-rated and higher-budget movies tend to perform better at the box office. This correlation provides insights into predictive modeling, where sentiment analysis results can help forecast financial outcomes, aiding in decision-making for movie investments and marketing efforts. 
  
In summary, the analysis of IMDb movie reviews through data science techniques has highlighted the importance of understanding audience sentiment and its impact on financial performance. These insights can be instrumental in shaping marketing strategies, guiding film production decisions, and improving the overall success of movies in the market. If you need further analysis or have more visualizations to share, feel free to ask!  
  
