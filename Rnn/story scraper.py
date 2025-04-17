import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
import random

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import time

def scrape_story_selenium(url):
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    driver = webdriver.Chrome(options=options)
    driver.get(url)

    # Wait for content to load
    time.sleep(3)

    soup = BeautifulSoup(driver.page_source, "html.parser")
    driver.quit()

    block = soup.find("div", class_="block_panel")
    if block:
        None
    # Get all text inside .block_panel with line breaks on <br>
        story_text = block.get_text(separator="\n", strip=True)
        # Stop at "Info" if present
        stop_index = story_text.find("Read")
        # Set start index after "Introduction:"
        start_index = story_text.find("Introduction:")
        if start_index != -1:
            story_text = story_text[start_index:]
        if stop_index != -1:
            story_text = story_text[:stop_index]
        return story_text
    return None
BASE_URL = ""  
START_PAGE = BASE_URL + "/"  
OUTPUT_DIR = "stories"
os.makedirs(OUTPUT_DIR, exist_ok=True)



HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 ' +
                  '(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
    'Referer': 'https://www.google.com/',
    'Accept-Language': 'en-US,en;q=0.9',
}
VISITED = set()
STORY_URLS = set()
story_counter = 1

def is_same_domain(url):
    return urlparse(url).netloc == urlparse(BASE_URL).netloc

def get_story_links(url):
    story_links = set()
    try:
        resp = requests.get(url, headers=HEADERS)
        soup = BeautifulSoup(resp.text, 'html.parser')
        for a in soup.find_all('a', href=True):
            full_url = urljoin(BASE_URL, a['href'])
            if is_same_domain(full_url) and 'story' in full_url and full_url not in VISITED:
                story_links.add(full_url)
    except Exception as e:
        print(f"[!] Failed to get links from {url}: {e}")
    return story_links
def scrape_story(url):
    try:
        resp = requests.get(url, headers=HEADERS)
        soup = BeautifulSoup(resp.text, 'html.parser')

        blocks = soup.find_all('div', class_='block_panel')
        if not blocks:
            return None

        # Combine text from all block_panel divs
        story_parts = [block.get_text(separator="\n", strip=True) for block in blocks]
        full_story = "\n\n".join(story_parts).strip()
        # Stop at "Info" if present
        stop_index = full_story.find("Read")
        # Set start index after "Introduction:"
        start_index = full_story.find("Introduction:")
        if start_index != -1:
            full_story = full_story[start_index + len("Introduction:"):]
        if stop_index != -1:
            full_story = full_story[:stop_index]
        # Remove any unwanted text


        return full_story
    except Exception as e:
        print(f"[!] Failed to scrape {url}: {e}")
        return None
    
"""   
def scrape_story(url):
    try:
        resp = requests.get(url, headers=HEADERS)
        soup = BeautifulSoup(resp.text, 'html.parser')
        block = soup.find('div', class_='block_panel')
        if not block:
            return None

        # Get all text inside .block_panel with line breaks on <br>
        story_text = block.get_text(separator="\n", strip=True)
        # Stop at "Info" if present
        stop_index = story_text.find("Read")
        # Set start index after "Introduction:"
        start_index = story_text.find("Introduction:")
        if start_index != -1:
            story_text = story_text[start_index:]
        if stop_index != -1:
            story_text = story_text[:stop_index]
        return story_text
    except Exception as e:
        print(f"[!] Failed to scrape {url}: {e}")
        return None
"""

to_visit = {START_PAGE}


while to_visit:
    current = to_visit.pop()
    if current in VISITED:
        continue

    session = requests.Session()
    session.headers.update(HEADERS)
    response = session.get(current)

    VISITED.add(current)
    print(f"[+] Visiting: {current}")
    story_links = get_story_links(current)

    for link in story_links:
        if link not in STORY_URLS:
            STORY_URLS.add(link)
            story_text = scrape_story(link)
            
            if story_text:
                file_name = f"story_{story_counter}.txt"
                file_path = os.path.join(OUTPUT_DIR, file_name)
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(story_text)
                print(f"    [✓] Saved: {file_name}")
                story_counter += 1

    to_visit.update(story_links)


