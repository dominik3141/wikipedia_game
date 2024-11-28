import openai
import os
import json
import argparse
import requests
from bs4 import BeautifulSoup
from flask import Flask, render_template, request, Response, jsonify
import time
from typing import Iterator
from datetime import datetime
from urllib.parse import unquote, quote
import pytz
import logging
import configparser
from pathlib import Path

app = Flask(__name__)
app.config['APPLICATION_ROOT'] = '/wikipedia-game'


def load_api_key() -> str:
    """Loads OpenAI API key from config file"""
    config = configparser.ConfigParser()
    config.read('api_key.conf')
    return config['DEFAULT']['OPENAI_API_KEY']

openai.api_key = load_api_key()
openai_model = "gpt-4o-mini"

SYSTEM_PROMPT = """
You are a helpful assistant playing the Wikipedia game.
Your goal is to find the shortest path between two Wikipedia pages by selecting links that get you closest to the target page.

Given a current Wikipedia page and a target page, select the most promising link that will get you closer to the target.

IMPORTANT: Respond only with a JSON object in this format:
{
    "url": "full_wikipedia_url",
    "link_name": "name_of_the_link"
}
"""

# Add this before the logging configuration
os.environ['TZ'] = 'America/Los_Angeles'
time.tzset()  # Only works on Unix-like systems

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S %Z',
    handlers=[
        logging.FileHandler('wikipedia_game.log'),
        logging.StreamHandler()
    ]
)

# Configure timezone for all handlers
logger = logging.getLogger()
for handler in logger.handlers:
    handler.setFormatter(logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s',
        '%Y-%m-%d %H:%M:%S %Z'
    ))
    handler.converter = lambda *args: datetime.now(pytz.timezone('America/Los_Angeles')).timetuple()


def call_openai_api(current_page: str, target_page: str, valid_links: list[dict[str, str]], attempt: int = 0) -> dict:
    """Makes an API call to get the next link choice from valid options, with retries"""
    max_attempts = 3
    try:
        # Add valid links to the prompt with explicit instructions
        links_text = "\n".join([f"- {link['link_name']}" for link in valid_links])
        
        # Make the constraints more explicit if this is a retry
        constraint_text = ""
        if attempt > 0:
            constraint_text = "\nIMPORTANT: Your previous selection was invalid. You MUST choose from the exact links provided above."
        
        prompt = f"""Current page: {format_wiki_url(current_page)}
Target page: {format_wiki_url(target_page)}

Available links (select the one most likely to lead to the target):
{links_text}{constraint_text}"""

        response = openai.chat.completions.create(
            model=openai_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        content = response.choices[0].message.content

        # Clean up the response
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:].strip()

        next_move = json.loads(content)
        
        # Verify the selected link is valid
        matching_link = next((link for link in valid_links 
                            if link['link_name'] == next_move['link_name']), None)
        
        if not matching_link:
            if attempt < max_attempts:
                log_game_progress(
                    f"Invalid link selected: '{next_move['link_name']}', retrying (attempt {attempt + 1}/{max_attempts})", 
                    "WARNING"
                )
                return call_openai_api(current_page, target_page, valid_links, attempt + 1)
            else:
                raise ValueError(f"Selected link '{next_move['link_name']}' is not in the list of valid links after max retries")
        
        # Use the URL from our valid links to ensure correctness
        next_move['url'] = matching_link['url']
        return next_move

    except Exception as e:
        if attempt < max_attempts:
            log_game_progress(
                f"API call failed, retrying (attempt {attempt + 1}/{max_attempts})", 
                "WARNING"
            )
            return call_openai_api(current_page, target_page, valid_links, attempt + 1)
            
        error_context = {
            "error_type": type(e).__name__,
            "error_message": str(e),
            "current_page": format_wiki_url(current_page),
            "target_page": format_wiki_url(target_page),
            "attempts": attempt + 1
        }
        print(f"API call failed after {attempt + 1} attempts: {json.dumps(error_context, indent=2)}")
        return None


def format_wiki_url(url: str) -> str:
    """Formats a Wikipedia URL to show just the article name"""
    try:
        article_name = url.split("/wiki/")[-1]
        return unquote(article_name).replace("_", " ")
    except:
        return url


def log_game_progress(message: str, level: str = "INFO") -> None:
    """Logs messages to file with PST timestamp"""
    level_map = {
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "SUCCESS": logging.INFO,
        "FAILED": logging.WARNING
    }
    logging.log(level_map.get(level, logging.INFO), message)


def generate_events(start: str, end: str, max_steps: int) -> Iterator[str]:
    """Streams the Wikipedia game progress as SSE events"""
    log_game_progress(
        f"New game request: {format_wiki_url(start)} → {format_wiki_url(end)} (max {max_steps} steps)"
    )
    
    current_page = start
    steps = 0

    while current_page != end and steps < max_steps:
        yield f"data: {json.dumps({'type': 'thinking'})}\n\n"

        response = requests.get(current_page)
        _, valid_links = get_wikipedia_content(response.text)
        
        if not valid_links:
            error_msg = f"No valid links found on page: {format_wiki_url(current_page)}"
            log_game_progress(error_msg, "ERROR")
            yield f"data: {json.dumps({'type': 'error', 'message': error_msg})}\n\n"
            return

        next_move = call_openai_api(current_page, end, valid_links)
        
        if not next_move:
            error_msg = f"Failed to get next move from {format_wiki_url(current_page)} to {format_wiki_url(end)} at step {steps}"
            log_game_progress(error_msg, "ERROR")
            yield f"data: {json.dumps({'type': 'error', 'message': error_msg})}\n\n"
            return

        current_page = next_move["url"]
        steps += 1

        log_game_progress(
            f"Step {steps}: Following link '{next_move['link_name']}' → {format_wiki_url(current_page)}"
        )

        yield f"data: {json.dumps({'type': 'step', 'step': steps, 'link_name': next_move['link_name'], 'url': current_page})}\n\n"

    success = current_page == end
    message = "Success! Target reached!" if success else "Failed to reach target within step limit"
    status = "SUCCESS" if success else "FAILED"
    log_game_progress(
        f"Game complete: {message} (took {steps} steps)", 
        status
    )
    
    yield f"data: {json.dumps({'type': 'complete', 'message': message})}\n\n"


def get_wikipedia_site(url: str) -> str:
    """Returns the main content of a Wikipedia page"""
    response = requests.get(url)
    return get_wikipedia_content(response.text)


def get_wikipedia_content(html: str) -> tuple[str, list[dict[str, str]]]:
    """Extracts the main article content and valid links from a Wikipedia page HTML"""
    soup = BeautifulSoup(html, "html.parser")
    content = soup.find(id="mw-content-text")
    
    if not content:
        return "", []

    # Extract all valid Wikipedia links
    valid_links = []
    for link in content.find_all('a'):
        href = link.get('href', '')
        # Only include internal Wikipedia links that aren't special pages
        if (href.startswith('/wiki/') 
            and ':' not in href  # Excludes special pages like Category:, File:, etc
            and not href.startswith('/wiki/Main_Page')):
            valid_links.append({
                'url': f"https://en.wikipedia.org{href}",
                'link_name': link.get_text()
            })

    # Remove unwanted elements for text content
    for unwanted in content.find_all(["table", "script", "style", "sup", ".mw-editsection"]):
        unwanted.decompose()

    # Extract text content
    paragraphs = []
    for p in content.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6"]):
        text = p.get_text().strip()
        if text:
            paragraphs.append(text)

    return "\n\n".join(paragraphs), valid_links


def search_wikipedia(query: str) -> list[dict[str, str]]:
    """Searches Wikipedia and returns matching articles"""
    if not query or len(query.strip()) < 3:
        return []
        
    search_url = f"https://en.wikipedia.org/w/api.php?action=opensearch&search={quote(query)}&limit=5&namespace=0&format=json"
    try:
        response = requests.get(search_url)
        data = response.json()
        # API returns [query, [titles], [descriptions], [urls]]
        return [
            {"title": title, "url": url}
            for title, url in zip(data[1], data[3])
        ]
    except Exception as e:
        log_game_progress(f"Search failed: {str(e)}", "ERROR")
        return []


@app.route("/stream")
def stream() -> Response:
    """Handles the SSE stream endpoint"""
    start = request.args.get("start")
    end = request.args.get("end")
    max_steps = int(request.args.get("max_steps", 10))

    return Response(
        generate_events(start, end, max_steps), mimetype="text/event-stream"
    )


@app.route("/")
def index() -> str:
    """Serves the main page"""
    return render_template("index.html")


@app.route("/search")
def search():
    """Endpoint for Wikipedia article search"""
    query = request.args.get("q", "")
    if not query or len(query.strip()) < 3:
        return jsonify([])
        
    search_url = (
        "https://en.wikipedia.org/w/api.php?"
        "action=opensearch"
        "&format=json"
        "&search=" + quote(query) +
        "&limit=5"
        "&namespace=0"
    )
    
    try:
        response = requests.get(search_url)
        data = response.json()
        results = [
            {"title": title, "url": url}
            for title, url in zip(data[1], data[3])
        ]
        return jsonify(results)
    except Exception as e:
        print(f"Search failed: {str(e)}")  # For debugging
        return jsonify([])


if __name__ == "__main__":
    app.run(debug=True)
