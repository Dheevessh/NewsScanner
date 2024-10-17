from flask import Flask, jsonify, render_template, request
import requests
from transformers import pipeline
from bs4 import BeautifulSoup
import spacy
import en_core_web_sm
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer

# Initialize the Flask app
app = Flask(__name__)

# Load NLP Model for text processing (spaCy)
nlp = en_core_web_sm.load()

# Load the summarization model for abstractive summarization
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Your News API Key (replace this with your actual API key)
NEWS_API_KEY = '9ae45e3cf5e643d2b7d6cade4ae0aff3'  # Get this from https://newsapi.org/

def get_news():
    """Fetch news headlines from News API."""
    url = f"https://newsapi.org/v2/top-headlines?country=us&apiKey={NEWS_API_KEY}"
    response = requests.get(url)
    news_data = response.json()
    articles = news_data.get('articles', [])
    return articles

def scrape_article(url):
    """Scrape text from a given URL."""
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')

        # Try to find the main content area using different strategies
        # You can add or change these as necessary based on the actual HTML structure
        article_body = soup.find('div', class_='article__body')  # Adjust as per actual class name
        if article_body is None:
            article_body = soup.find('div', class_='entry-content')  # Alternative class name
        if article_body is None:
            article_body = soup.find('div', class_='content')  # Another possible class

        if not article_body:
            return "Could not scrape article content."

        # Extract paragraphs from the identified content area
        paragraphs = article_body.find_all('p')
        article_text = " ".join([p.get_text() for p in paragraphs])
        
        # Return the combined text of the article
        return article_text.strip()
    
    except Exception as e:
        return f"Error scraping article: {str(e)}"




def preprocess_text(text):
    """Preprocess text using spaCy: Tokenization, removing stop words, and lemmatization."""
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens)

def summarize_news_abstractive(text):
    """Use AI to summarize the news article with abstractive summarization (transformers)."""
    try:
        # Calculate dynamic max_length, ensuring it's not longer than the input
        input_length = len(text.split())
        max_length = min(60, input_length)  # Set max_length to be the input length or 150, whichever is smaller
        min_length = max(5, max_length // 2)  # Set min_length as half of max_length, but at least 60

        # Generate a summary using AI
        summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        return f"Error generating summary: {str(e)}"

def extractive_summary(text):
    """Summarize text using extractive LSA summarization (sumy)."""
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, 4)  # 4 sentences
    return " ".join([str(sentence) for sentence in summary])

@app.route('/api/news')
def news():
    """Fetch news and summarize each article."""
    articles = get_news()
    categorized_articles = {}

    for article in articles:
        title = article.get('title', 'No Title')
        url = article.get('url', '#')
        category = article.get('source', {}).get('name', 'General')  # Use source as category

        # Scrape the full article content
        article_text = scrape_article(url)
        
        # Check if article_text was successfully scraped
        if article_text and len(article_text) > 100:  # Ensure sufficient text length for summarization
            # Preprocess the article text
            processed_text = preprocess_text(article_text)
            # Abstractive summary by default
            summary_text = summarize_news_abstractive(processed_text)
        else:
            summary_text = "Could not scrape sufficient article content."

        # Add to category
        if category not in categorized_articles:
            categorized_articles[category] = []
        
        categorized_articles[category].append({
            'title': title,
            'summary': summary_text,
            'url': url
        })

    return jsonify(categorized_articles)


@app.route('/summarize_url', methods=['POST'])
def summarize_url():
    """Scrape a URL, process it, and summarize it."""
    url = request.form.get('url')
    summary_type = request.form.get('summary_type', 'abstractive')  # Choose summarization type (default is abstractive)

    # Scrape article content
    article_text = scrape_article(url)

    # Preprocess article text
    processed_text = preprocess_text(article_text)

    # Summarize based on the selected type
    if summary_type == 'extractive':
        summary = extractive_summary(processed_text)
    else:
        summary = summarize_news_abstractive(processed_text)

    return jsonify({'summary': summary})

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
