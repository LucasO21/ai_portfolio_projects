# Libraries ----
import os
from pprint import pprint
import json
import re
import requests
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pandas as pd
from bs4 import BeautifulSoup


from langchain_community.document_loaders import FireCrawlLoader
from firecrawl import FirecrawlApp
from firecrawl.types import ScrapeOptions

from src.global_utilities.keys import get_env_key
from src.global_utilities.llms import get_llm


# Define API Keys ----
FIRECRAWL_API_KEY = get_env_key("firecrawl")
OPENAI_API_KEY = get_env_key("openai")


# Initialize Firecrawl App ----
firecrawl_app = FirecrawlApp(api_key=FIRECRAWL_API_KEY)


# ------------------------------------------------------------------------------
# TESTING MIXCLOUD SCRAPE (1 DJ)----
# ------------------------------------------------------------------------------

# https://www.mixcloud.com/djeflosa/
# https://www.mixcloud.com/djsprenk/

# Helper Functions ----
def parse_relative_date(text: str):
    # Match number + unit (d, w, mo, y)
    match = re.match(r"(\d+)(d|w|mo|m|y) ago", text.strip())
    if not match:
        raise ValueError(f"Invalid format: {text}")

    value, unit = int(match[1]), match[2]
    now = datetime.now()

    if unit == "d":
        return (now - timedelta(days=value)).date()
    elif unit == "w":
        return (now - timedelta(weeks=value)).date()
    elif unit == "mo" or unit == "m":
        return (now - relativedelta(months=value)).date()
    elif unit == "y":
        return (now - relativedelta(years=value)).date()


# DJ URL ----
url = "https://www.mixcloud.com/djeflosa/"

# Scrape Page ----
result = firecrawl_app.scrape(
    url=url,
    formats=["html"]
)

print(result)

# Soup ----
soup = BeautifulSoup(result.html, "html.parser")

# Remove Unwanted Tags ----
for tag in soup.find_all("path"):
    tag.decompose()

for tag in soup.find_all("img"):
    tag.decompose()

soup_trimmed = str(soup)

# Step 1: Extract DJ Info ----

# - Name ----
dj_name = soup.find_all("h1", class_ = "styles__DisplayTitle-css-in-js__sc-go2u8s-3 kOjGCU")[0].text.strip()

# - Following Count ----
following_count = soup.find_all("span", class_ = "button__StyledChildren-css-in-js__sc-1hu2thj-1 eLSuGE")[0].text.strip().split()[0]
following_count = int(following_count)

# - Followers Count ----
followers_count = soup.find_all("span", class_ = "button__StyledChildren-css-in-js__sc-1hu2thj-1 eLSuGE")[1].text.strip().split()[0].replace(",", "")
followers_count = int(followers_count)

# - Location ----
location = soup.find_all("p", class_ = "Location__LocationText-css-in-js__sc-1qywssu-2 kgpIhy")[0].text.strip()

# - Bio ----
bio_tag = soup.find_all("div", class_ = "styles__Text-css-in-js__sc-3bsl01-4 OQOaz")
bio = " ".join(p.get_text(strip=True) for p in bio_tag)

# - Show URLs ----
all_links = [a["href"] for a in soup.find_all("a", href=True)]

# - Social Links ----
social_links = [link for link in all_links if "facebook" in link or "instagram" in link or "patreon" in link or "soundcloud" in link]
social_links = list(dict.fromkeys(social_links))

# - Show URLs ----
name = re.escape(url.split("/")[-2])
pattern = rf"https://www\.mixcloud\.com/{name}/\d{{8}}"

show_links = [
    l for l in all_links if name in l and len(l) > len(url)
    and "/following/" not in l and "/followers/" not in l and "/hosts" not in l
    and "/reposts/" not in l and "/favorites/" not in l and "/listens/" not in l
]




# Extract All Data ----

# - Title ----
title = soup.find_all("h1", class_="styles__Title-css-in-js__sc-ozqgn7-9")[0].text.strip()

# - Play Count ----
play_count = int(soup.find_all("p", class_="styles__Label-css-in-js__sc-1yk6zpi-8")[0].text.strip().split(" ")[0].strip())

# - Likes Count ----
likes_count = int(soup.find_all("p", class_="styles__Label-css-in-js__sc-1yk6zpi-8")[1].text.strip().split(" ")[0].strip())

# - Posted Date ----
posted_date = soup.find_all("div", class_="styles__TimeSinceMobile-css-in-js__sc-1yk6zpi-6 exXVAM")[0].text.strip()
posted_date = parse_relative_date(posted_date).date()

# - Genre Tags ----
genre_tags = soup.find_all("span", class_="styles__GenreTagContainer-css-in-js__sc-1qdq7kd-0 ewEbCJ")
genre_tags = [tag.text.strip() for tag in genre_tags]

soup.find_all("span", id="L1")[0].text.strip().split("|")[0].strip()

# Set Info ----
other_set_info_tag = soup.find_all("div", class_="styles__Text-css-in-js__sc-3bsl01-4 cRCbfA")

# - Energy & Tempo ----
energy_tempo_span = [child.text.strip() for child in other_set_info_tag[0].find_all("span") if child.get("id") == "L1"]
min_energy, max_energy = map(int, energy_tempo_span[0].split("|")[0].replace("Energy","").split("-"))
min_tempo, max_tempo = map(int, energy_tempo_span[0].split("|")[1].replace("BPM","").split("-"))

# - Other Info ----
set_inspiration_list = [child.text.strip() for child in other_set_info_tag[0].find_all("span") if child.get("id") != "L1"]
set_inspiration =  "".join(set_inspiration_list)

# - Tracklist ----
tracklist_tag = soup.find_all("div", class_ = "styles__Paragraph-css-in-js__sc-12xxm55-1 VlRav")
tracklist = tracklist_tag[0].text.strip()