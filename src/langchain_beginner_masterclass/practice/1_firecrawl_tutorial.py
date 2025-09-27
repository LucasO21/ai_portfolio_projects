# Libraries ----
import os
from pprint import pprint
import json
import re
import requests

from langchain_community.document_loaders import FireCrawlLoader
from firecrawl import FirecrawlApp, JsonConfig
from firecrawl import Firecrawl
from firecrawl.types import ScrapeOptions

from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import re
import pandas as pd

from src.global_utilities.keys import get_env_key
from src.global_utilities.llms import get_llm


# Define API Keys ----
FIRECRAWL_API_KEY = get_env_key("firecrawl")
OPENAI_API_KEY = get_env_key("openai")

firecrawl_app = FirecrawlApp(api_key=FIRECRAWL_API_KEY)

result = firecrawl_app.scrape_url(
    url="https://docs.firecrawl.dev"
)

print(result)

md = result["markdown"]

links = result["linksOnPage"]

metadata = result["metadata"]

result["metadata"]["title"]



# ------------------------------------------------------------------------------
# Cannondale Bikes Scrape ----
# ------------------------------------------------------------------------------

# - Homepage ----
url = "https://www.cannondale.com/en-us/bikes"

result = firecrawl_app.scrape_url(url=url)

result.keys()

md = result["markdown"]

links = result["linksOnPage"]

metadata = result["metadata"]


# Test Single Product Page ----
single_product_url = "https://www.cannondale.com/en-us/bikes/road/gravel/superx/superx-1"

single_product_result = firecrawl_app.scrape_url(url=single_product_url)

single_product_result.keys()

md = single_product_result["markdown"]

links = single_product_result["linksOnPage"]

metadata = single_product_result["metadata"]

# - Extract Section ----
def extract_section(text, section_name):
    # Regex to grab everything after ### {section_name} until the next ### (or end of text)
    pattern = rf"### {section_name}\n+([\s\S]*?)(?=\n### |\Z)"
    match = re.search(pattern, text, re.IGNORECASE)
    return match.group(1).strip() if match else None

model_description = extract_section(md, "Model Description")
pprint(model_description)
highlights = extract_section(md, "Highlights")

print(model_description)
print(highlights)


# - Extract Specs ----
def extract_specs(text, section_name):
    # Match everything under the section until the next ### or end
    section_pattern = rf"### {section_name}\n+([\s\S]*?)(?=\n### |\Z)"
    section_match = re.search(section_pattern, text)
    if not section_match:
        return {}

    section_text = section_match.group(1).strip()

    # Extract fields like "- **Frame**\n\nSuperX Carbon..."
    field_pattern = r"- \*\*(.+?)\*\*\n\n([\s\S]*?)(?=\n- \*\*|\Z)"
    fields = re.findall(field_pattern, section_text)

    return {name.strip(): value.strip() for name, value in fields}

details = extract_specs(md, "Details")
frameset = extract_specs(md, "Frameset")

print("DETAILS:", details, "\n")
print("FRAMESET:", frameset)



# ------------------------------------------------------------------------------
# Mixcloud Scrape ----
# ------------------------------------------------------------------------------
from pydantic import BaseModel
from bs4 import BeautifulSoup
import sys

url = "https://www.mixcloud.com/djsprenk/"

mixcloud_app = FirecrawlApp(api_key=FIRECRAWL_API_KEY)

class ExtractSchema(BaseModel):
    title: str


mixcloud_result = mixcloud_app.scrape(
    url = url,
    formats = ["html"]
)

html = mixcloud_result.html

soup = BeautifulSoup(html, "html.parser")

links = [a["href"] for a in soup.find_all("a", href=True)]


name = soup.find("h1", class_="styles__DisplayTitle-css-in-js__sc-go2u8s-3 kOjGCU").text
following = soup.find("span", class_="button__StyledChildren-css-in-js__sc-1hu2thj-1 eLSuGE").text.split()[0]

from langchain.text_splitter import RecursiveCharacterTextSplitter

mixcloud_result = mixcloud_app.scrape(
    url = url,
    formats = ["html", "markdown"]
)

# Text 1 ----
md = mixcloud_result.markdown

# - HTML ----
html = mixcloud_result.html

soup = BeautifulSoup(html, "html.parser")
soup_trimmed = re.sub(r'd=M 0 35.*?35(?=\s|>|$)', 'd=M 0 35', str(soup), flags=re.DOTALL)


# - Splitter ----
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ".", " ", ""]
)

chunks = splitter.split_text(md)
len(chunks)

# - Metadata ----
docs = [
    {
        "page_content": chunk,
        "metadata": {
            "source": url,
            "name": name,
            "following": following,
            # "followers": followers,
            "url": url,
            "links": links
        }
    }
    for chunk in chunks
]

# - Embeddings ----
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY)

# - Vector Store ----
from langchain_community.vectorstores import Chroma

vectorstore = Chroma.from_documents(
    documents = docs,
    embedding = embeddings,
    persist_directory = "src/langchain_beginner_masterclass/practice/database/mixcloud_1",
    collection_name = "mixcloud_collection"
)


import re
import json
from datetime import datetime
from typing import List, Dict, Any

def extract_dj_info(text: str) -> Dict[str, Any]:
    """Extract DJ profile information"""

    # Extract basic info
    name_match = re.search(r'# (DJ \w+)', text)
    location_match = re.search(r'(Cambridge, United States)', text)
    followers_match = re.search(r'([\d,]+) Followers', text)
    following_match = re.search(r'([\d,]+) Following', text)

    # Extract bio - everything between the location and "Show more"
    bio_pattern = r'Cambridge, United States\n\n(.*?)(?:Show more|$)'
    bio_match = re.search(bio_pattern, text, re.DOTALL)


    dj_info = {}

    if name_match:
        dj_info['name'] = name_match.group(1)

    if location_match:
        dj_info['location'] = location_match.group(1)

    if followers_match:
        dj_info['followers'] = int(followers_match.group(1).replace(',', ''))

    if following_match:
        dj_info['following'] = int(following_match.group(1).replace(',', ''))

    if bio_match:
        bio = bio_match.group(1).strip()
        # Clean up the bio text
        bio = re.sub(r'\n+', ' ', bio)  # Replace multiple newlines with space
        dj_info['bio'] = bio



    return dj_info

dj_info = extract_dj_info(md)



# - dj name ----
url = "https://www.mixcloud.com/djsprenk/"
name = url.split("/")[-2]

# - Extract All Links ----
def extract_https_links(text):
    """Extract all HTTPS URLs from the given text."""
    # Pattern to match https URLs
    https_pattern = r'https://[^\s\)\]\}"]+'

    # Find all matches
    links = re.findall(https_pattern, text)

    # Remove duplicates while preserving order
    unique_links = list(dict.fromkeys(links))

    return unique_links

# Usage:
links = extract_https_links(md)

# - social links ----
social_links = [
    link for link in links if "facebook" in link
    or "instagram" in link
    or "patreon" in link
    or "soundcloud" in link
]

# - valid show links ----
name = re.escape(name)
pattern = rf"https://www\.mixcloud\.com/{name}/\d{{8}}"

show_urls = []
for link in links:
    if re.match(pattern, link):
        show_urls.append(link)



single_show_url = show_urls[0]

# - scrape single show ----
single_show_result = mixcloud_app.scrape(
    url=single_show_url,
    formats = ["markdown"]
)

md_single_show = single_show_result.markdown

links = single_show_result["linksOnPage"]

metadata = single_show_result["metadata"]

dictionary = {
    "page_link": single_show_url,
    "page_content": md_single_show,
    "metadata": {
        "name": name,
        "dj_main_url": url,
        "following": following,

    }
}


dictionary = {
    "show_info": md_single_show,
}

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
import json

# Your source data
data = {
    'page_link': 'https://www.mixcloud.com/djsprenk/20250823-ariel-and-leticia-2/',
    'page_content': "[Home Link](https://www.mixcloud.com/)\n\n[Join Pro](https://www.mixcloud.com/pro/) Open Menu\n\n![Monalisa | Ariel & Leticia Lambada Reunion in Boston (Part 2)](https://thumbnailer.mixcloud.com/unsafe/290x290/extaudio/4/0/2/d/cf70-f086-40d4-a888-10a30dbbb7c3)\n\n[HQ](https://www.mixcloud.com/pro/)\n\n# Monalisa \\| Ariel & Leticia Lambada Reunion in Boston (Part 2)\n\n[281 plays](https://www.mixcloud.com/djsprenk/20250823-ariel-and-leticia-2/listeners/) [11 favorites](https://www.mixcloud.com/djsprenk/20250823-ariel-and-leticia-2/favorites/)\n2w ago\n\n2 weeks ago\n\nFavorite\n\nRepost\n\nShare\n\nAdd to\n\nMore\n\nFavorite\n\nRepost\n\nShare\n\nAdd to\n\n57:59\n\n![Avatar](<Base64-Image-Removed>)\n\n[![DJ Sprenk's profile picture](https://thumbnailer.mixcloud.com/unsafe/52x52/profile/0/e/f/2/5a55-2a54-45da-a41e-ef366fede9df)](https://www.mixcloud.com/djsprenk/)\n\n[**DJ Sprenk**](https://www.mixcloud.com/djsprenk/) [Pro User](https://www.mixcloud.com/pro/)\n\n[1,381 followers](https://www.mixcloud.com/djsprenk/followers/)\n\nFollow\n\n- [lambada](https://www.mixcloud.com/genres/lambada/)\n- [zouk lambada](https://www.mixcloud.com/genres/zouk-lambada/)\n- [zouk](https://www.mixcloud.com/genres/zouk/)\n- [brazilian zouk](https://www.mixcloud.com/genres/brazilian-zouk/)\n- [ghetto zouk](https://www.mixcloud.com/genres/ghetto-zouk/)\n\nEnergy 4-8 \\| 77-80 BPM\n\nSaturday night closer for the Ariel & Leticia Lambada Reunion by Lambazouk Boston.\n\nChapters: Romantic & intimate > flirty & energetic > relaxed & expansive > playful & upbeat.\n\nDJ Nerds: My second ever Lambada-focused set! This was a bit slower and more even energy than part 1 because it followed a long day of classes and a super high energy opener set. While I still wanted to keep energy & speed up with Lambada-able percussion throughout, I aimed instead to create differentiation through mood and acoustic qualities of each section of music.\n\nThanks again to Angela for organizing and trusting me to bring the Lambada!\n\nShow more\n\n### Chart history\n\nTop position held by this upload\n\n- [1stzouk](https://www.mixcloud.com/genres/zouk/)\n- [1stbrazilian zouk](https://www.mixcloud.com/genres/brazilian-zouk/)\n\n### Tracklist\n\nPlaying tracks by Kalipsxau, Mika Mendes, Nelson Freitas, Elji Beatzkilla, Beyoncé feat. Sean Paul and more.\n\n## Comments\n\n![Avatar](<Base64-Image-Removed>)",
    'metadata': {
        'name': 'djsprenk',
        'dj_main_url': 'https://www.mixcloud.com/djsprenk/',
        'following': '81'
    }
}

# - embeddings ----
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY)

# - text splitter ----
document = Document(
    page_content=data['page_content'],
    metadata={
        'page_link': data['page_link'],
        'name': data['metadata']['name'],
        'dj_main_url': data['metadata']['dj_main_url'],
        'following': data['metadata']['following']
    }
)

# Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # Size of each chunk
    chunk_overlap=50,  # Overlap between chunks
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
)

# Split the document into chunks
chunks = text_splitter.split_documents([document])

print(f"Document split into {len(chunks)} chunks")
print("First chunk preview:")
print(f"Content: {chunks[0].page_content[:200]}...")
print(f"Metadata: {chunks[0].metadata}")

# - vector store ----
# Create the Chroma vector store
print("Creating Chroma vector store...")

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="src/langchain_beginner_masterclass/practice/database/mixcloud_1"
)

print(f"Vector store created with {len(chunks)} documents!")
print("Vector store saved to './chroma_db' directory")

# Test the vector store with a simple query
print("\nTesting vector store with a sample query...")
test_query = "Lambada music"
results = vectorstore.similarity_search(test_query, k=2)

print(f"Query: '{test_query}'")
print(f"Found {len(results)} similar documents:")
for i, result in enumerate(results):
    print(f"\nResult {i+1}:")
    print(f"Content preview: {result.page_content[:150]}...")
    print(f"Metadata: {result.metadata}")


retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}  # Retrieve top 3 most relevant chunks
)


from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Test the retriever
print("Testing retriever...")
test_docs = retriever.get_relevant_documents("What music genres does DJ Sprenk play?")
print(f"Retriever found {len(test_docs)} relevant documents")

llm = get_llm("openai", "gpt-4o", OPENAI_API_KEY)

prompt_template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer based on the context, just say that you don't know.

Context: {context}

Question: {question}
Answer:"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# Create the RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": PROMPT},
    return_source_documents=True
)

result = qa_chain({"query": "What is the name of the DJ?"})
pprint(result)

result = qa_chain({"query": "What occasion was this set for?"})
pprint(result)

result = qa_chain({"query": "What was the energy level of the set?"})
pprint(result)


result = qa_chain({"query": "how may times has this set been played and favorited?"})
pprint(result)


result = qa_chain({"query": "Give me a short profile of this set based on the context?"})
pprint(result)

#! Helpers ----
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


# Show URL HTML ----
single_show_result_html = mixcloud_app.scrape(
    url=single_show_url,
    formats = ["html"]
)

soup = BeautifulSoup(single_show_result_html.html, "html.parser")

for tag in soup.find_all("path"):
    tag.decompose()

for tag in soup.find_all("img"):
    tag.decompose()

html = str(soup)

html

# - View All Tags ----
for tag in soup.find_all():
    print(tag.name)
    print(tag.attrs)
    print(tag.text)
    print("\n")

# - View All h1 Tags & Classes ----
type_of_tag = "l1"

for tag in soup.find_all(type_of_tag):
    print(tag.text)
    print(tag.attrs)
    print("\n")


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


# ------------------------------------------------------------------------------
# BATCH SCRAPE ALL SHOWS ----
# ------------------------------------------------------------------------------

batch_show_urls = show_urls[0:3]

batch_scrape_results = mixcloud_app.start_batch_scrape(
    urls = batch_show_urls,
    formats = ["html"]
)

print(batch_scrape_results)

batch_scrape_status = mixcloud_app.get_batch_scrape_status(batch_scrape_results.id)
print(batch_scrape_status)


# Batch Scrape Results ----
batch_html = batch_scrape_status.data

html_list = []
for html in batch_html:
    html_list.append(html.html)

html_list

# - Beautiful Soup ----
soup_list = []
for html in html_list:
    bs = BeautifulSoup(html, "html.parser")

    # - remove unwanted tags ----
    for tag in bs.find_all("path"):
        tag.decompose()

    for tag in bs.find_all("img"):
        tag.decompose()

    soup_list.append(bs)


# - Extract All Data ----
title_list = []
play_count_list = []
likes_count_list = []
posted_date_list = []
tags_list = []

for soup in soup_list:
    title = soup.find_all("h1", class_="styles__Title-css-in-js__sc-ozqgn7-9")[0].text.strip()
    title_list.append(title)

    play_count = int(soup.find_all("p", class_="styles__Label-css-in-js__sc-1yk6zpi-8")[0].text.strip().split(" ")[0].strip())
    play_count_list.append(play_count)

    likes_count = int(soup.find_all("p", class_="styles__Label-css-in-js__sc-1yk6zpi-8")[1].text.strip().split(" ")[0].strip())
    likes_count_list.append(likes_count)

    posted_date = soup.find_all("div", class_="styles__TimeSinceMobile-css-in-js__sc-1yk6zpi-6 exXVAM")[0].text.strip()
    posted_date = parse_relative_date(posted_date)
    posted_date_list.append(posted_date)

    tags = soup.find_all("li", class_="styles__GenreTagListItem-css-in-js__sc-j82gfl-2 hXBWie")
    tags = [tag.text.strip() for tag in tags]
    tags_list.append(tags)



# - Create DataFrame ----
df = pd.DataFrame({
    "title": title_list,
    "play_count": play_count_list,
    "likes_count": likes_count_list,
    "posted_date": posted_date_list
})

print(parse_relative_date("1mo ago"))

parse_relative_date("1m ago")