

#Import Modules
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis, linregress
import statsmodels.api as sm
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

kld_list = pd.read_csv('KLDscores.csv')
metadata_list = pd.read_csv('SPGC-metadata-2018-07-18.csv')

# Prepare metadata_list

# text preprocessing
def preprocess_text(text):
    text = text.lower()
    # Remove punctuation using regex
    text = re.sub(r'[^\w\s]', '', text)
    return text

# Apply preprocessing to book description column
df = pd.DataFrame()
df['id'] = metadata_list['id']
df['downloads'] = metadata_list['downloads']
df['processed_description'] = metadata_list['subjects'].apply(preprocess_text)

# Initializing TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')

# Fitting and transforming the processed descriptions
tfidf_matrix = tfidf_vectorizer.fit_transform(df['processed_description'])

# classifying books into genres

genre_keywords = {
    'Fiction': [
        'imagination', 'narrative', 'characters', 'plot', 'story', 'literature', 'prose', 'fantasy', 'romance',
        'adventure', 'thriller', 'suspense', 'mystery', 'fiction', 'novel', 'drama', 'epic', 'saga', 'allegory',
        'myth', 'fable', 'novella', 'storytelling', 'character-driven', 'plot-driven', 'short story', 'fictional',
        'sea stories', 'domestic', 'political', 'legal thriller', 'science fiction', 'utopian',
        'dystopian', 'war stories', 'love stories', 'satire', 'gothic', 'pastoral', 'tragedies', 'didactic',
        'bildungsromans', 'humorous', 'ghost', 'fairy tales', 'folklore', 'paradoxical', 'paranormal'
    ],
    'Non-Fiction': [
        'factual', 'real', 'informative', 'documentary', 'true story', 'educational', 'biography', 'memoir', 'essay',
        'reference', 'analysis', 'research', 'non-fiction', 'guide', 'manual', 'how-to', 'textbook', 'instructional',
        'scholarly', 'report', 'non-fictional', 'journal', 'chronicle', 'historical account', 'autobiography',
        'essays', 'poetry', 'epics', 'handbooks', 'manuals', 'sources', 'speeches', 'addresses', 'political science',
        'geography', 'methodology'
    ],
    'Historical': [
        'history', 'period', 'past', 'era', 'heritage', 'historical events', 'authenticity', 'ancient', 'medieval',
        'reconstruction', 'civilization', 'chronicle', 'historical', 'antiquity', 'archaeological', 'cultural',
        'traditional', 'historic', 'bygone', 'retrospective', 'historical narrative', 'historiography', 'timeline',
        'historical fiction', 'revolution', 'civil war', 'colonial', 'new plymouth', 'foreign relations',
        'population statistics', 'political history', 'constitutional history', 'inaugural addresses', 'census'
    ],
    'Science': [
        'research', 'discovery', 'experiment', 'theory', 'data', 'analysis', 'biology', 'chemistry', 'physics',
        'innovation', 'technology', 'scientific', 'exploration', 'technical', 'medical', 'environmental', 'engineering',
        'biological', 'chemical', 'physical', 'scientific discovery', 'technological advancement', 'experimental',
        'laboratory', 'scientific method', 'internet', 'computer networks', 'information retrieval', 'data processing',
        'electronic publishing', 'text processing', 'computer security', 'radio broadcasting', 'mathematical constants',
        'number theory'
    ],
    'Other': [
        'self-help', 'motivation', 'improvement', 'success', 'advice', 'wellness', 'personal growth', 'mental health',
        'habits', 'coaching', 'guidance', 'humor', 'poetry', 'drama', 'art', 'philosophy', 'self-development',
        'self-improvement', 'motivational', 'inspirational', 'self-care', 'self-love', 'mindfulness', 'meditation',
        'spirituality', 'leadership', 'creativity', 'satire', 'parody', 'comedic', 'artistic', 'philosophical',
        'reflective', 'children\'s literature', 'young adult', 'juvenile', 'fairy tales', 'sacred books', 'bible',
        'christian', 'mythology', 'folklore', 'epic poetry', 'greek drama', 'drama adaptations', 'magical realism',
        'urban fantasy', 'paranormal romance', 'imaginary places', 'mythological characters'
    ]
}

# Function to assign genre based on keywords
def assign_genre(top_keywords):
    for genre, keywords in genre_keywords.items():
        if any(keyword in top_keywords for keyword in keywords):
            return genre
    return 'Other'

# Number of top keywords to consider
num_top_keywords = 5

# Iterating over each book's subject
for i, desc in enumerate(df['processed_description']):
    tfidf_scores = tfidf_matrix[i].toarray().flatten()
    top_keyword_indices = tfidf_scores.argsort()[-num_top_keywords:][::-1]
    top_keywords = [tfidf_vectorizer.get_feature_names_out()[idx] for idx in top_keyword_indices]
    genre = assign_genre(top_keywords)

    # Assign genre to new column
    df.loc[i, 'genre'] = genre

kld_list['kld_values'] = kld_list['kld_values'].apply(eval)

# Create a DataFrame to store the book-level measures
book_measures = pd.DataFrame()
# Calculate average and variance of KLD scores
book_measures['id'] = kld_list['filename']
book_measures['min_kld'] = kld_list['kld_values'].apply(np.min)
book_measures['max_kld'] = kld_list['kld_values'].apply(np.max)
book_measures['avg_kld'] = kld_list['kld_values'].apply(np.mean)
book_measures['std_dev_kld'] = kld_list['kld_values'].apply(np.std)
book_measures['range_kld'] = kld_list['kld_values'].apply(np.max) - kld_list['kld_values'].apply(np.min)

# Function to compute the slope of a linear regression
def compute_slope(kld_values):
    x = np.arange(len(kld_values))

    slope, _, _, _, _ = linregress(x, kld_values)
    return slope

# Calculate the slope of KLD scores over the course of the narrative
book_measures['slope_kld'] = kld_list['kld_values'].apply(compute_slope)

#Combining the data
df = pd.merge(df,kld_list['filename'], left_on='id', right_on='filename', how='inner')
df = df.drop('filename', axis=1)

# combining book measures also
df = pd.merge(df,book_measures, on= 'id', how = 'inner' )

# Obtaining half the value of the min non-zero values in downloads
no_zeros_df = df[df['downloads'] != 0]
c = min(no_zeros_df['downloads']) * 0.5

# replace downloads (with downloads + c) to avoid error from log transformation
df['downloads_new'] = df['downloads'] + 0.5

X = df[['avg_kld', 'std_dev_kld', 'slope_kld', 'range_kld']]
X = sm.add_constant(X)
y = np.log(df['downloads_new'])



# Fitting model
model = sm.OLS(y, X).fit()
print(model.summary())

#creating dummies in for the genre
df_with_dummies = pd.get_dummies(df, columns=['genre'], drop_first=True)

df_with_dummies.head(5)

# cleaning up dummies
df_with_dummies['genre_Historical'] = np.where(df['genre'] == 'genre_Historical', 1, 0)
df_with_dummies['genre_Non-Fiction'] = np.where(df['genre'] == 'genre_Non-Fiction', 1, 0)
df_with_dummies['genre_Others'] = np.where(df['genre'] == 'genre_Others', 1, 0)
df_with_dummies['genre_Science'] = np.where(df['genre'] == 'genre_Science', 1, 0)
df_with_dummies['genre_Other'] = np.where(df['genre'] == 'genre_Other', 1, 0)

df_with_dummies.head(5)

X = df_with_dummies.drop(['id', 'downloads', 'processed_description','min_kld','max_kld','downloads_new'], axis=1)

#scaling data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#fitting lasso model


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

lasso = Lasso(alpha=0.2)
lasso.fit(X_train, y_train)

#model Evaluation
y_pred = lasso.predict(X_test)

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"R-squared: {r2}")
print(f"Mean Squared Error: {mse}")

#more information
coefficients = pd.DataFrame({
    'Variable': X.columns,
    'Coefficient': lasso.coef_
})

print(coefficients)

#Linear regression with gender
A = df_with_dummies[['avg_kld', 'std_dev_kld', 'slope_kld',	'genre_Historical','genre_Science'	, 'genre_Non-Fiction', 'genre_Others',	'genre_Other']]
A = sm.add_constant(X)
b = np.log(df_with_dummies['downloads_new'])

# Fitting model
model1 = sm.OLS(b, A).fit()
print(model1.summary())