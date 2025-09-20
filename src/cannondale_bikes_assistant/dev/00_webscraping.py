# Imports ----
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import time

# Setup Chrome options
chrome_options = Options()
chrome_options.add_argument("--headless")  # Run in background
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")
chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")

driver = webdriver.Chrome(options=chrome_options)

# Constants ----
URL = "https://www.cannondale.com/en-us/bikes"

driver.get(URL)

# Wait for page to load
WebDriverWait(driver, 10).until(
    EC.presence_of_element_located((By.CLASS_NAME, "product-card card filter-and-sort__product -has-3Q "))
)

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}


# Scrape the page ----
response = requests.get(URL, headers=headers)
soup = BeautifulSoup(driver.page_source, "html.parser")

print(soup.prettify())

# Find all the bike links ----
bike_links = soup.find_all("div", class_="product-card card filter-and-sort__product -has-3Q ")

