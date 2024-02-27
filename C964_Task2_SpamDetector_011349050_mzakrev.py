import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, precision_score
from scipy.sparse import hstack
from sklearn.svm import SVC
from wordcloud import WordCloud, STOPWORDS
import tkinter as tk
from tkinter import scrolledtext
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# get our preprocessed csv file from the relative path
#encode it using ISO in accordance with most excel based CSV
file_path = r'datasets/electronics_b_quotations.csv'
df = pd.read_csv(file_path, encoding="ISO-8859-1")




#create a UI window in the form of an application
#uses the tkinter import and labels it
#adjust size so that window is a minimum of 1024x768
app = tk.Tk()
app.title("Product Review Analysis")
app.geometry("1024x768")

#function to get the length of a users review, will auto calculate for them
def get_review_length(user_input):
    # Calculate length of user input or placeholder text
    return len(user_input) if user_input != "Enter your review here..." else 0

#create a function to analyze a user's input review
#it takes the users input, converts it using tfidf
#considers the review_text, rating, likes, and length of review
#then outputs whether it thinks the review is Spam or not
def analyze_review():
    user_input = review_entry.get("1.0", "end-1c")
    rating_value = float(rating_entry.get())
    likes_value = int(likes_entry.get())
    review_length = get_review_length(user_input)

    user_input_tfidf = vectorizer.transform([user_input])
    user_input_final = hstack([[likes_value, rating_value, review_length], user_input_tfidf])

    prediction = model.predict(user_input_final)[0]
    probability = model.predict_proba(user_input_final)[0][1]  # Probability of being fake

    result_label.config(text=f"Prediction: {'Spam' if prediction == 1 else 'Non-Spam'}, Probability: {round(probability, 2)}")

#our function to show graphs when a button with the correct tag is pressed
#the 3 graphs are bar, scatter, and word cloud
#these are connected with the labels below
#defines the colors and numbers (1 or 0) to denote spam vs non-spam
def show_graph(graph_type):
    if graph_type == 'Bar':
        plt.figure(figsize=(8, 6))
        sns.countplot(x='is_spam_1', data=df)
        plt.title('Distribution of Spam and Non-Spam Reviews')
        plt.xlabel('Non-Spam (0) or Is Spam (1)')
        plt.ylabel('Count')
        plt.show()
    elif graph_type == 'Scatter':
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x='liked', y='rating', hue='is_spam_1', data=df)
        plt.title('Scatter Plot of Likes and Ratings')
        plt.xlabel('Likes')
        plt.ylabel('Rating')
        plt.xlim(0, 10000)
        plt.legend(title='BLUE=Non-Spam (0) or ORANGE=Is Spam (1)')
        plt.show()
    elif graph_type == 'WordCloud':
        spam_reviews = df.loc[df['is_spam_1'] == 1, 'review_text'].values
        spam_reviews_text = ' '.join(map(str, spam_reviews))

        wordcloud = WordCloud(width=800, height=400, max_words=200, background_color='black', stopwords=stopwords).generate(spam_reviews_text)

        plt.figure(figsize=(10, 8))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Word Cloud for Spam Reviews')
        plt.show()

# button to display accuracy, precision, and classification report
#uses the same data as our print to console metrics
def display_metrics():
    y_pred = model.predict(X_test_final)

    # Handle UndefinedMetricWarning
    precision = precision_score(y_test, y_pred, zero_division=1)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    metrics_text = f'Accuracy: {accuracy:.2f}\n\nPrecision: {precision:.2f}\n\nClassification Report:\n{report}'
    metrics_label.config(text=metrics_text)

#UI components to include buttons for all 3 graphs
#user_input to analyze a review and necessary labels
#includes the position, the padding, and the filler data
review_label = ttk.Label(app, text="Enter Product Review:")
review_label.pack(pady=10)

review_entry = scrolledtext.ScrolledText(app, wrap=tk.WORD, width=40, height=5)
review_entry.pack(pady=10)

rating_label = ttk.Label(app, text="Enter Rating:")
rating_label.pack(pady=5)

rating_entry = ttk.Entry(app)
rating_entry.pack(pady=5)

likes_label = ttk.Label(app, text="Enter Number of Likes:")
likes_label.pack(pady=5)

likes_entry = ttk.Entry(app)
likes_entry.pack(pady=5)

#Placeholder text for the user_input boxes, graphs and metrics
review_entry.insert("1.0", "Enter your review here...")
rating_entry.insert(0, "5.0")
likes_entry.insert(0, "100")

analyze_button = ttk.Button(app, text="Analyze Review", command=analyze_review)
analyze_button.pack(pady=10)

result_label = ttk.Label(app, text="")
result_label.pack(pady=10)

graph_buttons_frame = ttk.Frame(app)
graph_buttons_frame.pack(pady=10)

bar_button = ttk.Button(graph_buttons_frame, text="Bar Graph", command=lambda: show_graph('Bar'))
bar_button.pack(side=tk.LEFT, padx=5)

scatter_button = ttk.Button(graph_buttons_frame, text="Scatter Plot", command=lambda: show_graph('Scatter'))
scatter_button.pack(side=tk.LEFT, padx=5)

wordcloud_button = ttk.Button(graph_buttons_frame, text="Word Cloud", command=lambda: show_graph('WordCloud'))
wordcloud_button.pack(side=tk.LEFT, padx=5)

metrics_button = ttk.Button(app, text="Display Metrics", command=display_metrics)
metrics_button.pack(pady=10)

metrics_label = ttk.Label(app, text="")
metrics_label.pack(pady=10)

# separate text and non-text features
#identifies which columns the model will be trained on
X_text = df['review_text']
X_non_text = df[['liked', 'rating', 'review_length']]
y = df['is_spam_1']


# split the preprocessed data into training and test sets
X_train_text, X_test_text, X_train_non_text, X_test_non_text, y_train, y_test = train_test_split(
    X_text, X_non_text, y, test_size=0.2, random_state=42)

# TF-IDF vectorization for text features
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train_text)
X_test_tfidf = vectorizer.transform(X_test_text)

# combine TF-IDF features with non-text features
X_train_final = hstack([X_train_non_text.values, X_train_tfidf])
X_test_final = hstack([X_test_non_text.values, X_test_tfidf])

# train our model based on Logistic Regression
model = LogisticRegression(max_iter=2000)
#model = SVC() #use this to run SVM model #placeholder for SVM
model.fit(X_train_final, y_train)
y_pred = model.predict(X_test_final)

# handle UndefinedMetricWarning, in the case of no input or missing data in CSV
precision = precision_score(y_test, y_pred, zero_division=1)

#defines our accuracy and precision scores, then prints them to console
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy:.2f}\n')
print(f'Precision: {precision:.2f}\n')
print('Classification Report:\n', report)

#stop words that are used in nearly every review to not appear on heatmap
#convert all words to lowercase to correctly identify whether it is the same word
#these words were chosen, as they may not be relevant to all reviews, or too common
custom_stopwords = set(["one", "use", "used", "work", "kindle", "nook", "camera", "will",
                        "radio", "well", "computer", "tv", ".", "speaker", "problem"])
custom_stopwords_lower = set(word.lower() for word in custom_stopwords)
stopwords_lower = set(STOPWORDS)
stopwords = stopwords_lower.union(custom_stopwords_lower)

#start the main loop application
app.mainloop()
