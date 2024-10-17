from flask import Flask, jsonify, render_template
import requests
from transformers import pipeline

app = Flask(__name__)

# AI model for summarization
summarizer = pipeline("summarization", model="path/to/local/model")


# Your News API Key
NEWS_API_KEY = '9ae45e3cf5e643d2b7d6cade4ae0aff3'  # Get this from https://newsapi.org/

def get_news():
    """Fetch news headlines from News API."""
    url = f"https://newsapi.org/v2/top-headlines?country=us&apiKey={NEWS_API_KEY}"
    response = requests.get(url)
    news_data = response.json()
    articles = news_data.get('articles', [])
    return articles

def summarize_news(text):
    """Use AI to summarize the news article."""
    summary = summarizer(text, max_length=50, min_length=25, do_sample=False)
    return summary[0]['summary_text']

@app.route('/api/news')
def news():
    """Fetch news and summarize them."""
    articles = get_news()
    summarized_articles = []

    for article in articles:
        title = article.get('title', 'No Title')
        description = article.get('description', '')
        url = article.get('url', '#')

        # Summarize the description using AI
        summary = summarize_news(description) if description else "No description available."
        
        summarized_articles.append({
            'title': title,
            'summary': summary,
            'url': url
        })
    
    return jsonify(summarized_articles)

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
