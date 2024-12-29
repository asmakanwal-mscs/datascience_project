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

**Text reviews:** The actual movie reviews written by users.

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
(5043, 28)
# Display summary statistics
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
![image](https://github.com/user-attachments/assets/4550b4e3-038d-4f88-a370-6a152dc46327)

The bar graph shows the count of movies per genre. Key Insights: X-axis (Genre): Represents the movie genres. Y-axis (Count): Shows the number of movies in each genre. Bar Height: Taller bars indicate more movies in that genre. Rotation: Genre labels are rotated for better readability. the bar graph visualizes the distribution of movies across different genres.
# 3.Bar Graph: Count of Movies Per Genre
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
The heatmap shows correlations between numeric variables:

Color Intensity: Indicates the strength of the correlation (red for positive, blue for negative). Values: Correlation coefficients range from -1 to 1. Annotations: Display the exact correlation values. It highlights relationships between variables in the dataset.

# 4.Heatmap: Correlation Between Numeric Variables
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
Key Insights: X-axis (IMDb Score): Represents the IMDb score range. Y-axis (Frequency): Shows how many movies fall within each IMDb score range. KDE (Kernel Density Estimate): The smooth curve shows the overall distribution of scores. Bins: The number of bins (20) helps visualize the distribution in more detail. In summary, the histogram helps understand the spread and concentration of IMDb scores across movies, highlighting common score ranges and any skewness.
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
Insights from Applied ML Models

1. Regression Model (IMDb Score Prediction) Key Features: Budget, gross earnings, user reviews, and duration. Performance: RMSE quantifies prediction error. Example: RMSE = 0.85 indicates average deviation of 0.85 IMDb points. Feature Importance: User reviews and gross earnings likely contribute the most to predictions. Limitations: Subjective factors like audience sentiment aren't captured, and missing/skewed data affects accuracy.

2. Classification Model (Content Rating Prediction) Key Features: Budget, gross earnings, user reviews. Performance: Example accuracy = 82%. Precision and recall reveal strengths/weaknesses for each rating. Feature Importance: Budget and user reviews help distinguish ratings. Limitations: Imbalanced data and subjective criteria (e.g., violence) impact predictions.

General Takeaways Random Forest models provide interpretable predictions but are sensitive to data quality. Enhancements: Include richer features (e.g., genres, director), and optimize hyperparameters for better accuracy.


[ ]
![image](https://github.com/user-attachments/assets/c3a80b5e-d2ef-43d7-bc93-3d3a27b7d3e6)
![image](https://github.com/user-attachments/assets/92aef67c-a5e1-4705-8515-ae3247f1808a)
![image](https://github.com/user-attachments/assets/afcfbcc7-dc03-4d82-a9d7-b3174e09ed0c)
![image](https://github.com/user-attachments/assets/c0ff34a8-2201-449f-96cb-bff43a954b7d)
Machine Learning Integration Steps: Regression for IMDb score prediction. Classification for content rating prediction. Evaluate models using RMSE, accuracy, and classification reports. Visualize results for better insight.

