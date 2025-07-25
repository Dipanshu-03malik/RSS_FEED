import os # cite: 1
import logging # cite: 1, 2
from datetime import datetime, timedelta # cite: 1, 2, 3, 4
import schedule # cite: 2
import time # cite: 2
import feedparser # cite: 5
import requests # cite: 5
from typing import Dict, Any, List # cite: 4, 5
from tqdm import tqdm # cite: 5
from langdetect import detect, LangDetectException # cite: 5
import pandas as pd # cite: 3, 4, 5
from bs4 import BeautifulSoup # cite: 4, 5
import re # cite: 4, 5
import streamlit as st # cite: 3
import glob # cite: 3

# --- Directory Setup ---
# Ensure directories exist before logging is configured
os.makedirs('data', exist_ok=True) # cite: 1
os.makedirs('logs', exist_ok=True) # cite: 1

# --- Global Logging Configuration (applies to all modules after this point) ---
logging.basicConfig( # cite: 1, 2, 4
    level=logging.INFO, # cite: 1, 2, 4
    format='%(asctime)s - %(levelname)s - %(message)s', # cite: 1, 2, 4
    handlers=[ # cite: 1, 2
        logging.FileHandler(f'logs/application_{datetime.now().strftime("%Y%m%d")}.log'), # Combined log file # cite: 1, 2
        logging.StreamHandler() # cite: 1, 2
    ]
)
logger = logging.getLogger(__name__) # cite: 1, 2, 4

# --- RSS Feeds Data (from rss_links.py) ---
RSS_FEEDS = [ # cite: 6
    # India
    {
        "url": "https://timesofindia.indiatimes.com/rssfeeds/-2128936835.cms",
        "agency": "Times of India",
        "country": "India"
    },
    {
        "url": "https://www.thehindu.com/news/national/feeder/default.rss",
        "agency": "The Hindu",
        "country": "India"
    },
    {
        "url": "https://indianexpress.com/section/india/feed/",
        "agency": "Indian Express",
        "country": "India"
    },
    {
        "url": "https://www.ndtv.com/feeds/latest",
        "agency": "NDTV",
        "country": "India"
    },
    {
        "url": "https://www.hindustantimes.com/feeds/rss",
        "agency": "Hindustan Times",
        "country": "India"
    },
    
    # USA
    {
        "url": "http://rss.cnn.com/rss/edition.rss",
        "agency": "CNN",
        "country": "USA"
    },
    {
        "url": "https://feeds.npr.org/1001/rss.xml",
        "agency": "NPR",
        "country": "USA"
    },
    {
        "url": "https://www.nytimes.com/services/xml/rss/nyt/HomePage.xml",
        "agency": "New York Times",
        "country": "USA"
    },
    {
        "url": "https://www.washingtonpost.com/news-sitemap-index.xml",
        "agency": "Washington Post",
        "country": "USA"
    },
    {
        "url": "https://www.foxnews.com/about/rss/",
        "agency": "Fox News",
        "country": "USA"
    },
    
    # UK
    {
        "url": "http://feeds.bbci.co.uk/news/rss.xml",
        "agency": "BBC News",
        "country": "UK"
    },
    {
        "url": "https://www.theguardian.com/uk/rss",
        "agency": "The Guardian",
        "country": "UK"
    },
    {
        "url": "https://www.telegraph.co.uk/rss.xml",
        "agency": "The Telegraph",
        "country": "UK"
    },
    {
        "url": "https://www.independent.co.uk/rss",
        "agency": "The Independent",
        "country": "UK"
    },
    
    # Japan
    {
        "url": "https://www3.nhk.or.jp/rss/news/cat0.xml",
        "agency": "NHK",
        "country": "Japan"
    },
    
    # Germany
    {
        "url": "https://rss.dw.com/xml/rss-de-all",
        "agency": "Deutsche Welle",
        "country": "Germany"
    },
    
    # France
    {
        "url": "https://www.lemonde.fr/rss/une.xml",
        "agency": "Le Monde",
        "country": "France"
    },
    {
        "url": "https://www.lefigaro.fr/rss/figaro_actualites.xml",
        "agency": "Le Figaro",
        "country": "France"
    },
    {
        "url": "https://www.liberation.fr/rss/",
        "agency": "Libération",
        "country": "France"
    },
    
    # Canada
    {
        "url": "https://www.cbc.ca/cmlink/rss-topstories",
        "agency": "CBC News",
        "country": "Canada"
    },
    
    # Australia
    {
        "url": "https://www.abc.net.au/news/feed/51120/rss.xml",
        "agency": "ABC News",
        "country": "Australia"
    },
    {
        "url": "https://www.smh.com.au/rss/feed.xml",
        "agency": "Sydney Morning Herald",
        "country": "Australia"
    },
    {
        "url": "https://www.theaustralian.com.au/feed/",
        "agency": "The Australian",
        "country": "Australia"
    },
    
    # Russia
    {
        "url": "https://tass.com/rss/v2.xml",
        "agency": "TASS",
        "country": "Russia"
    },
    
    # South Korea
    {
        "url": "https://www.koreatimes.co.kr/www/rss/world.xml",
        "agency": "Korea Times",
        "country": "South Korea"
    },
    
    # Malaysia
    {
        "url": "https://www.thestar.com.my/rss/editors-choice",
        "agency": "The Star",
        "country": "Malaysia"
    },
    
    # Singapore
    {
        "url": "https://www.straitstimes.com/news/world/rss.xml",
        "agency": "The Straits Times",
        "country": "Singapore"
    },
    
    # Indonesia
    {
        "url": "https://www.antaranews.com/rss/terkini",
        "agency": "Antara News",
        "country": "Indonesia"
    },
    
    # Brazil
    {
        "url": "https://g1.globo.com/dynamo/rss2.xml",
        "agency": "G1",
        "country": "Brazil"
    },
    
    # Mexico
    {
        "url": "https://www.mexiconewsdaily.com/feed/",
        "agency": "Mexico News Daily",
        "country": "Mexico"
    },
    
    # UAE
    {
        "url": "https://www.emirates247.com/rss",
        "agency": "Emirates 24/7",
        "country": "UAE"
    },
    {
        "url": "https://gulfnews.com/rss",
        "agency": "Gulf News",
        "country": "UAE"
    },
    {
        "url": "https://www.arabianbusiness.com/rss",
        "agency": "Arabian Business",
        "country": "UAE"
    },
    
    # Italy
    {
        "url": "https://www.ansa.it/sito/ansait_rss.xml",
        "agency": "ANSA",
        "country": "Italy"
    },
    
    # South Africa
    {
        "url": "https://www.businesslive.co.za/rss",
        "agency": "Business Live",
        "country": "South Africa"
    },
    
    # New Zealand
    {
        "url": "https://www.stuff.co.nz/rss",
        "agency": "Stuff",
        "country": "New Zealand"
    }
]

# --- Utility Functions (from utils.py) ---
def clean_text(text: str) -> str: # cite: 4
    """
    Clean and normalize text content.
    """
    if not text: # cite: 4
        return "" # cite: 4
    
    # Remove HTML tags
    soup = BeautifulSoup(text, 'html.parser') # cite: 4
    text = soup.get_text() # cite: 4
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text) # cite: 4
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,!?-]', '', text) # cite: 4
    
    return text.strip() # cite: 4

def parse_date(date_str: str) -> datetime: # cite: 4
    """
    Parse various date formats to datetime object.
    Returns a timezone-naive datetime object.
    """
    if not date_str: # cite: 4
        return datetime.now() # cite: 4
        
    try: # cite: 4
        # Try parsing with pandas (handles most common formats)
        dt = pd.to_datetime(date_str) # cite: 4
        # Convert to naive datetime
        if dt.tzinfo is not None: # cite: 4
            dt = dt.tz_localize(None) # cite: 4
        return dt # cite: 4
    except: # cite: 4
        try: # cite: 4
            # Try parsing with datetime (ISO format with timezone)
            dt = datetime.strptime(date_str, '%Y-%m-%dT%H:%M:%S%z') # cite: 4
            return dt.replace(tzinfo=None) # cite: 4
        except: # cite: 4
            try: # cite: 4
                # Try parsing without timezone
                dt = datetime.strptime(date_str, '%Y-%m-%dT%H:%M:%S') # cite: 4
                return dt # cite: 4
            except: # cite: 4
                try: # cite: 4
                    # Try parsing just the date
                    dt = datetime.strptime(date_str, '%Y-%m-%d') # cite: 4
                    return dt # cite: 4
                except: # cite: 4
                    logger.warning(f"Could not parse date: {date_str}") # cite: 4
                    return datetime.now() # cite: 4

def is_within_timeframe(date: datetime, days: int = 365) -> bool: # cite: 4
    """
    Check if the date is within the specified timeframe.
    Handles offset-naive and offset-aware datetime comparison.
    """
    cutoff_date = datetime.now() # cite: 4
    # Convert both to naive UTC for comparison
    if date.tzinfo is not None: # cite: 4
        date = date.replace(tzinfo=None) # cite: 4
    if cutoff_date.tzinfo is not None: # cite: 4
        cutoff_date = cutoff_date.replace(tzinfo=None) # cite: 4
    cutoff_date = cutoff_date - timedelta(days=days) # cite: 4
    return date >= cutoff_date # cite: 4

def deduplicate_entries(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]: # cite: 4
    """
    Remove duplicate entries based on title and URL.
    """
    seen = set() # cite: 4
    unique_entries = [] # cite: 4
    
    for entry in entries: # cite: 4
        # Create a unique identifier using title and URL
        identifier = f"{entry['title']}_{entry['url']}" # cite: 4
        
        if identifier not in seen: # cite: 4
            seen.add(identifier) # cite: 4
            unique_entries.append(entry) # cite: 4
    
    return unique_entries # cite: 4

def save_to_csv(data: List[Dict[str, Any]], filename: str) -> None: # cite: 4
    """
    Save data to CSV file.
    """
    try: # cite: 4
        df = pd.DataFrame(data) # cite: 4
        df.to_csv(filename, index=False, encoding='utf-8') # cite: 4
        logger.info(f"Successfully saved data to {filename}") # cite: 4
    except Exception as e: # cite: 4
        logger.error(f"Error saving data to {filename}: {str(e)}") # cite: 4

def save_to_json(data: List[Dict[str, Any]], filename: str) -> None: # cite: 4
    """
    Save data to JSON file.
    """
    try: # cite: 4
        df = pd.DataFrame(data) # cite: 4
        df.to_json(filename, orient='records', lines=True) # cite: 4
        logger.info(f"Successfully saved data to {filename}") # cite: 4
    except Exception as e: # cite: 4
        logger.error(f"Error saving data to {filename}: {str(e)}") # cite: 4

# --- RSS Scraper Class (from scraper.py) ---
class RSSScraper: # cite: 5
    def __init__(self, feed_info: Dict[str, str]): # cite: 5
        self.url = feed_info['url'] # cite: 5
        self.agency = feed_info['agency'] # cite: 5
        self.country = feed_info['country'] # cite: 5
        self.headers = { # cite: 5
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36' # cite: 5
        }

    def detect_language(self, text: str) -> str: # cite: 5
        """
        Detect the language of the given text.
        """
        try: # cite: 5
            return detect(text) # cite: 5
        except LangDetectException: # cite: 5
            return 'unknown' # cite: 5

    def fetch_feed(self) -> List[Dict[str, Any]]: # cite: 5
        """
        Fetch and parse RSS feed.
        
        Returns:
            List of dictionaries containing parsed feed entries
        """
        try: # cite: 5
            # Add delay to respect rate limits
            time.sleep(1) # cite: 5
            
            # Fetch feed with custom headers
            response = requests.get(self.url, headers=self.headers, timeout=30) # cite: 5
            response.raise_for_status() # cite: 5
            
            # Parse feed
            feed = feedparser.parse(response.content) # cite: 5
            
            if feed.bozo: # cite: 5
                logger.warning(f"Feed parsing issues for {self.url}: {feed.bozo_exception}") # cite: 5
            
            entries = [] # cite: 5
            for entry in feed.entries: # cite: 5
                try: # cite: 5
                    # Extract and clean data
                    title = clean_text(entry.get('title', '')) # cite: 5
                    
                    # Try different possible fields for description/summary
                    description = '' # cite: 5
                    if 'description' in entry: # cite: 5
                        description = clean_text(entry['description']) # cite: 5
                    elif 'summary' in entry: # cite: 5
                        description = clean_text(entry['summary']) # cite: 5
                    elif 'content' in entry: # cite: 5
                        description = clean_text(entry['content'][0]['value']) # cite: 5
                    
                    # Get the article URL
                    link = entry.get('link', '') # cite: 5
                    
                    # Parse publication date
                    published = parse_date(entry.get('published', '')) # cite: 5
                    if published.tzinfo is not None: # cite: 5
                        published = published.replace(tzinfo=None) # cite: 5
                    
                    # Only include entries within the last year
                    if not is_within_timeframe(published): # cite: 5
                        continue # cite: 5
                    
                    # Detect language from title and description
                    combined_text = f"{title} {description}" # cite: 5
                    language = self.detect_language(combined_text) # cite: 5
                    
                    entries.append({ # cite: 5
                        'title': title, # cite: 5
                        'description': description, # cite: 5
                        'url': link, # cite: 5
                        'published_date': published.isoformat(), # cite: 5
                        'source': self.agency, # cite: 5
                        'country': self.country, # cite: 5
                        'language': language # cite: 5
                    })
                except Exception as e: # cite: 5
                    logger.error(f"Error processing entry from {self.url}: {str(e)}") # cite: 5
                    continue # cite: 5
            
            return deduplicate_entries(entries) # cite: 5
            
        except requests.RequestException as e: # cite: 5
            logger.error(f"Error fetching feed {self.url}: {str(e)}") # cite: 5
            return [] # cite: 5
        except Exception as e: # cite: 5
            logger.error(f"Unexpected error processing {self.url}: {str(e)}") # cite: 5
            return [] # cite: 5

    def fetch_historical_feed(self, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]: # cite: 5
        """Fetch historical feed data between start_date and end_date"""
        all_entries = [] # cite: 5
        current_date = start_date # cite: 5
        
        while current_date <= end_date: # cite: 5
            try: # cite: 5
                # Add date parameter to feed URL if supported
                date_str = current_date.strftime('%Y-%m-%d') # cite: 5
                feed_url = f"{self.url}?date={date_str}" # cite: 5
                
                # Fetch feed with rate limiting
                time.sleep(2)  # Rate limiting # cite: 5
                feed = feedparser.parse(feed_url, request_headers=self.headers) # cite: 5
                
                if feed.entries: # cite: 5
                    for entry in feed.entries: # cite: 5
                        entry_date = self._parse_date(entry.get('published', '')) # cite: 5
                        if start_date <= entry_date <= end_date: # cite: 5
                            all_entries.append(self._process_entry(entry)) # cite: 5
                
                current_date += timedelta(days=1) # cite: 5
                
            except Exception as e: # cite: 5
                logging.error(f"Error fetching historical feed for {date_str}: {str(e)}") # cite: 5
                continue # cite: 5
                
        return all_entries # cite: 5
    
    def _parse_date(self, date_str: str) -> datetime: # cite: 5
        """Parse date string to datetime object"""
        try: # cite: 5
            return pd.to_datetime(date_str).to_pydatetime() # cite: 5
        except: # cite: 5
            return datetime.now() # cite: 5
    
    def _process_entry(self, entry: Dict[str, Any]) -> Dict[str, Any]: # cite: 5
        """Process a single feed entry"""
        # Try different possible fields for description/summary
        description = '' # cite: 5
        if 'description' in entry: # cite: 5
            description = self._clean_text(entry['description']) # cite: 5
        elif 'summary' in entry: # cite: 5
            description = self._clean_text(entry['summary']) # cite: 5
        elif 'content' in entry: # cite: 5
            description = self._clean_text(entry['content'][0]['value']) # cite: 5
        
        return { # cite: 5
            'title': self._clean_text(entry.get('title', '')), # cite: 5
            'description': description, # cite: 5
            'url': entry.get('link', ''), # cite: 5
            'published_date': self._parse_date(entry.get('published', '')).isoformat(), # cite: 5
            'source': self.agency, # cite: 5
            'country': self.country # cite: 5
        }
    
    def _clean_text(self, text: str) -> str: # cite: 5
        """Clean HTML and special characters from text"""
        if not text: # cite: 5
            return '' # cite: 5
        # Remove HTML tags
        soup = BeautifulSoup(text, 'html.parser') # cite: 5
        text = soup.get_text() # cite: 5
        # Remove special characters
        text = re.sub(r'[^\w\s.,!?-]', '', text) # cite: 5
        return text.strip() # cite: 5

def scrape_all_feeds(feeds: List[Dict[str, str]], output_format: str = 'csv') -> None: # cite: 5
    """
    Scrape all provided RSS feeds and save the results.
    
    Args:
        feeds: List of feed information dictionaries
        output_format: Output format ('csv' or 'json')
    """
    all_entries = [] # cite: 5
    
    # Create progress bar
    for feed_info in tqdm(feeds, desc="Scraping feeds"): # cite: 5
        scraper = RSSScraper(feed_info) # cite: 5
        entries = scraper.fetch_feed() # cite: 5
        all_entries.extend(entries) # cite: 5
        
        # Save individual feed data
        if entries: # cite: 5
            filename = f"data/{feed_info['country']}_{feed_info['agency']}_{datetime.now().strftime('%Y%m%d')}.{output_format}" # cite: 5
            if output_format == 'csv': # cite: 5
                save_to_csv(entries, filename) # cite: 5
            else: # cite: 5
                save_to_json(entries, filename) # cite: 5
    
    # Save combined data
    if all_entries: # cite: 5
        combined_filename = f"data/all_news_{datetime.now().strftime('%Y%m%d')}.{output_format}" # cite: 5
        if output_format == 'csv': # cite: 5
            save_to_csv(all_entries, combined_filename) # cite: 5
        else: # cite: 5
            save_to_json(all_entries, combined_filename) # cite: 5
        
        logger.info(f"Total entries collected: {len(all_entries)}") # cite: 5

# --- Main Scraper Execution Function (adapted from main.py) ---
def run_scraper_main(): # cite: 1
    try: # cite: 1
        logger.info("Starting RSS feed scraping...") # cite: 1
        logger.info(f"Total feeds to process: {len(RSS_FEEDS)}") # cite: 1
        
        # Scrape all feeds and save as CSV
        scrape_all_feeds(RSS_FEEDS, output_format='csv') # cite: 1
        
        logger.info("RSS feed scraping completed successfully!") # cite: 1
        
    except Exception as e: # cite: 1
        logger.error(f"An error occurred: {str(e)}") # cite: 1
        raise # cite: 1

# --- Scheduler Functions (from scheduler.py) ---
def scheduled_job(): # cite: 2
    """
    Job function to run the scraper.
    """
    try: # cite: 2
        logger.info("Starting scheduled scraping job...") # cite: 2
        run_scraper_main() # Call the scraper main function # cite: 2
        logger.info("Scheduled scraping job completed successfully!") # cite: 2
    except Exception as e: # cite: 2
        logger.error(f"Error in scheduled job: {str(e)}") # cite: 2

def run_scheduler(): # cite: 2
    """
    Main function to set up and run the scheduler.
    """
    # Schedule the job to run every 6 hours
    schedule.every(6).hours.do(scheduled_job) # cite: 2
    
    # Run the job immediately on startup
    scheduled_job() # cite: 2
    
    logger.info("Scheduler started. Running every 6 hours.") # cite: 2
    
    # Keep the script running
    while True: # cite: 2
        schedule.run_pending() # cite: 2
        time.sleep(60) # cite: 2

# --- Streamlit App (from streamlit_app.py) ---
# Set page config
st.set_page_config( # cite: 3
    page_title="Global News Dashboard", # cite: 3
    page_icon="📰", # cite: 3
    layout="wide" # cite: 3
)

def load_data(): # cite: 3
    """
    Load and combine all news data from CSV files.
    """
    # Get all CSV files in the data directory
    csv_files = glob.glob('data/*.csv') # cite: 3
    
    if not csv_files: # cite: 3
        st.error("No data files found in the data directory!") # cite: 3
        return None # cite: 3
    
    # Read and combine all CSV files
    dfs = [] # cite: 3
    for file in csv_files: # cite: 3
        try: # cite: 3
            df = pd.read_csv(file) # cite: 3
            dfs.append(df) # cite: 3
        except Exception as e: # cite: 3
            st.warning(f"Error reading {file}: {str(e)}") # cite: 3
    
    if not dfs: # cite: 3
        st.error("No valid data found!") # cite: 3
        return None # cite: 3
    
    # Combine all dataframes
    combined_df = pd.concat(dfs, ignore_index=True) # cite: 3
    
    # Convert published_date to datetime
    try: # cite: 3
        # First, ensure the column exists
        if 'published_date' not in combined_df.columns: # cite: 3
            st.error("No published_date column found in the data!") # cite: 3
            return None # cite: 3
            
        # Convert to datetime, handling various formats
        combined_df['published_date'] = pd.to_datetime( # cite: 3
            combined_df['published_date'], # cite: 3
            errors='coerce' # cite: 3
        )
        
        # Remove rows with invalid dates
        combined_df = combined_df.dropna(subset=['published_date']) # cite: 3
        
        if len(combined_df) == 0: # cite: 3
            st.error("No valid dates found in the data!") # cite: 3
            return None # cite: 3
            
    except Exception as e: # cite: 3
        st.error(f"Error parsing dates: {str(e)}") # cite: 3
        return None # cite: 3
    
    # Remove duplicates
    combined_df = combined_df.drop_duplicates(subset=['title', 'url']) # cite: 3
    
    return combined_df # cite: 3

def run_streamlit_app(): # New function to encapsulate Streamlit app
    """
    Main function to run the Streamlit app.
    """
    st.title("📰 Global News Dashboard") # cite: 3
    
    # Load data
    df = load_data() # cite: 3
    if df is None: # cite: 3
        return # cite: 3
    
    # Sidebar filters
    st.sidebar.header("Filters") # cite: 3
    
    # Country filter
    countries = sorted(df['country'].unique()) # cite: 3
    selected_countries = st.sidebar.multiselect( # cite: 3
        "Select Countries", # cite: 3
        countries, # cite: 3
        default=countries[:5] # cite: 3
    )
    
    # Source filter
    sources = sorted(df['source'].unique()) # cite: 3
    selected_sources = st.sidebar.multiselect( # cite: 3
        "Select Sources", # cite: 3
        sources, # cite: 3
        default=sources[:5] # cite: 3
    )
    
    # Date range filter
    min_date = df['published_date'].min().date() # cite: 3
    max_date = df['published_date'].max().date() # cite: 3
    date_range = st.sidebar.date_input( # cite: 3
        "Select Date Range", # cite: 3
        value=(max_date - timedelta(days=7), max_date), # cite: 3
        min_value=min_date, # cite: 3
        max_value=max_date # cite: 3
    )
    
    # Language filter
    languages = sorted(df['language'].unique()) # cite: 3
    selected_languages = st.sidebar.multiselect( # cite: 3
        "Select Languages", # cite: 3
        languages, # cite: 3
        default=languages # cite: 3
    )
    
    # Apply filters
    mask = ( # cite: 3
        df['country'].isin(selected_countries) & # cite: 3
        df['source'].isin(selected_sources) & # cite: 3
        df['language'].isin(selected_languages) & # cite: 3
        (df['published_date'].dt.date >= date_range[0]) & # cite: 3
        (df['published_date'].dt.date <= date_range[1]) # cite: 3
    )
    filtered_df = df[mask] # cite: 3
    
    # Display statistics
    col1, col2 = st.columns(2) # cite: 3
    
    with col1: # cite: 3
        st.subheader("Articles by Country") # cite: 3
        country_counts = filtered_df['country'].value_counts() # cite: 3
        st.bar_chart(country_counts) # cite: 3
    
    with col2: # cite: 3
        st.subheader("Articles by Source") # cite: 3
        source_counts = filtered_df['source'].value_counts() # cite: 3
        st.bar_chart(source_counts) # cite: 3
    
    # Search functionality
    search_query = st.text_input("🔍 Search articles", "") # cite: 3
    if search_query: # cite: 3
        search_mask = ( # cite: 3
            filtered_df['title'].str.contains(search_query, case=False, na=False) | # cite: 3
            filtered_df['description'].str.contains(search_query, case=False, na=False) # cite: 3
        )
        filtered_df = filtered_df[search_mask] # cite: 3
    
    # Display articles table
    st.subheader(f"Articles ({len(filtered_df)})") # cite: 3
    
    # Format the dataframe for display
    display_df = filtered_df[['title', 'published_date', 'source', 'country', 'language']].copy() # cite: 3
    display_df['published_date'] = display_df['published_date'].dt.strftime('%Y-%m-%d %H:%M') # cite: 3
    
    # Add clickable links
    display_df['link'] = filtered_df['url'].apply( # cite: 3
        lambda x: f'<a href="{x}" target="_blank">Read Article</a>' # cite: 3
    )
    
    # Display the table with clickable links
    st.write( # cite: 3
        display_df.to_html(escape=False, index=False), # cite: 3
        unsafe_allow_html=True # cite: 3
    )

# --- Main Application Entry Point ---
if __name__ == "__main__":
    # Determine which part of the application to run based on an environment variable or command-line argument
    # For simplicity in a single file, we'll use an environment variable 'APP_MODE'
    # Example: To run the scheduler: APP_MODE=scheduler python your_single_file.py
    # Example: To run the Streamlit app: APP_MODE=streamlit streamlit run your_single_file.py
    # If no mode is specified, it could default to running the scraper once or the scheduler.
    
    app_mode = os.environ.get('APP_MODE', 'streamlit').lower() # Default to streamlit for easy testing

    if app_mode == 'scraper':
        logger.info("Running in SCRAPER mode.")
        run_scraper_main()
    elif app_mode == 'scheduler':
        logger.info("Running in SCHEDULER mode.")
        run_scheduler()
    elif app_mode == 'streamlit':
        logger.info("Running in STREAMLIT mode.")
        run_streamlit_app()
    else:
        logger.error(f"Invalid APP_MODE: {app_mode}. Please set APP_MODE to 'scraper', 'scheduler', or 'streamlit'.")