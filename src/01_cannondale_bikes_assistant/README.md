# 🚴‍♂️ Cannondale Bikes AI Assistant
***An Agentic RAG System with Tool-Calling and Conversational Memory***

<div align="center">
<img src="png/app_homepage.png" alt="Cannondale Bike AI Assistant Homepage" width="800">
</div>

---

## 📋 Project Overview

Shopping for a high-end bicycle online means sifting through hundreds of models, each with dense technical specifications spread across product pages. Comparing frames, drivetrains, and pricing across the Cannondale lineup is time-consuming and overwhelming, especially for riders who aren't sure what they need.

This project tackles that problem with an **agentic RAG (Retrieval-Augmented Generation) system** that acts like a knowledgeable bike shop employee. It uses an AI agent backed by five specialized tools to search, summarize, detail, compare, and recommend Cannondale bikes — all through natural conversation. The agent decides which tool to invoke based on the user's intent, retrieves relevant bike data from a MongoDB Atlas vector store, and synthesizes responses grounded in real product specifications scraped from [cannondale.com](https://www.cannondale.com/en-us).

What makes this implementation interesting is the **tool-calling agent architecture**: rather than a single RAG chain, the system routes queries through purpose-built tools, each with its own prompt template and retrieval strategy. This enables the assistant to handle everything from quick overviews to detailed side-by-side comparisons within a single conversational interface.

### 🎯 Key Concepts Explored

- **Agentic RAG Architecture**: An AI agent that autonomously selects from five specialized tools based on user intent, going beyond simple question-answering.
- **Tool-Calling with LangChain**: Defining structured tools with typed arguments (price ranges, bike types, experience levels) that the LLM invokes as function calls.
- **Vector Search**: Semantic similarity search over 326 bike specifications stored in MongoDB Atlas, with post-retrieval filtering by price and category.
- **Conversational Memory**: Multi-turn dialogue using Streamlit chat history, allowing natural follow-up questions like "what about the carbon version?" after an initial query.
- **Prompt Engineering**: Five distinct prompt templates optimized for different response modes — summaries, detailed specs, comparisons, recommendations, and search results.

---

## 🏗️ How It Works

The system uses an **agent-based architecture** where GPT-4o decides which specialized tool to use based on your question. Ask for a "quick summary" and it uses the summary tool. Ask to "compare the Synapse vs CAAD13" and it routes to the comparison tool. Each tool retrieves relevant bike data from MongoDB via vector similarity search, applies its own prompt template, and returns formatted results with bike images.

<div align="center">
<img src="png/how_it_works.png" alt="System Architecture" width="800">
</div>

### The Pipeline

1. **User Query**: The user asks a question in the Streamlit chat interface.
2. **Agent Decision**: GPT-4o analyzes the query and conversation history, then selects the appropriate tool — `search_bikes`, `get_bike_summary`, `get_bike_details`, `compare_bikes`, or `get_recommendation`.
3. **Vector Retrieval**: The chosen tool queries MongoDB Atlas using `text-embedding-ada-002` embeddings to find the most relevant bike documents via semantic similarity search.
4. **Post-Filtering**: Results are optionally filtered by bike type, price range, or budget constraints using structured tool arguments.
5. **Response Generation**: The retrieved context is fed through a tool-specific prompt template, and GPT-4o synthesizes a formatted answer with bike images and metadata.

### Five Specialized Tools

| Tool | Purpose | Key Features |
|------|---------|-------------|
| **Search Bikes** | Browse and filter the catalog | Filters by bike type, min/max price range |
| **Bike Summary** | Quick 3-4 sentence overview | Concise with bullet-point highlights |
| **Bike Details** | Full technical specifications | Comprehensive specs, components, metadata |
| **Compare Bikes** | Side-by-side comparison (2-3 bikes) | Structured tables with recommendations |
| **Get Recommendation** | Personalized suggestions | Considers budget, experience level, riding style |

### Conversational Follow-ups

Streamlit-based chat history maintains context across the session. Ask about a bike, then follow up with "how much does it cost?" or "compare it to the carbon version" without restating context.

<div align="center">
<img src="png/follow_up.png" alt="Conversational Follow-up Example" width="800">
</div>

---

## 🛠️ Tech Stack

```
🧠 LLM:         OpenAI GPT-4o
🔍 Embeddings:  OpenAI text-embedding-ada-002
🗄️ Data Store:  MongoDB Atlas (vector search)
🌐 Frontend:    Streamlit
🔗 Framework:   LangChain (Agent + Tools architecture)
📦 Other:       pymongo, python-dotenv, Poetry
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- Poetry (Python package manager)
- OpenAI API key
- MongoDB Atlas account (with vector search index enabled)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/LucasO21/ai_portfolio_projects.git
cd ai_portfolio_projects
```

2. **Install dependencies**
```bash
poetry install
```

3. **Set up environment variables**

Create a `.env` file in the project root:
```bash
OPENAI_API_KEY=your_openai_api_key_here
MONGO_DB_URI=your_mongodb_connection_string
```

4. **Prepare the vector store**

Load the bike data CSV into MongoDB Atlas and create embeddings:
```bash
# Run the vectorstore creation script (Jupyter notebook)
# See: src/01_cannondale_bikes_assistant/dev/01_create_vectorstore.py
```

Ensure your MongoDB Atlas cluster has a **vector search index** named `vector_index` on the `cannondale_bikes_db.bikes_collection` collection.

### Running the Application

```bash
poetry run streamlit run src/01_cannondale_bikes_assistant/app/app2.py
```

The app will be available at `http://localhost:8501`

---

## 💡 Example Usage

**Search & Filter:**
- "Show me mountain bikes under $5,000"
- "What gravel bikes do you have?"
- "List electric bikes between $4,000 and $8,000"

**Quick Summaries:**
- "Tell me about the Scalpel"
- "Quick summary of Synapse Carbon"

**Detailed Specs:**
- "Full specifications for Jekyll 1"
- "Detailed breakdown of SuperSix EVO"

**Comparisons:**
- "Compare Synapse vs CAAD13"
- "Differences between Topstone and Topstone Carbon"

**Recommendations:**
- "Best bike for weekend trail riding under $4,000"
- "What road bike for a beginner with $2,500 budget?"

---

## 📁 Project Structure

```
src/01_cannondale_bikes_assistant/
├── app/
│   ├── app2.py                  # Main Streamlit application (v2)
│   └── app.py                   # Legacy v1 application
├── dev/
│   ├── 01_create_vectorstore.py # Data ingestion & embedding pipeline
│   └── 02_rag_pipeline.py       # RAG experimentation notebook
├── database/
│   └── bikes_csv/
│       └── bikes_version_2.csv  # 326 Cannondale bikes scraped from cannondale.com
├── png/                         # Application screenshots
└── README.md
```

---

## 🔮 Future Improvements

- [ ] Add a sixth tool for retrieving bike reviews and rider feedback
- [ ] Implement multi-modal responses with side-by-side image comparisons
- [ ] Add session persistence so conversations survive browser refreshes
- [ ] Support additional bike brands beyond Cannondale

---

## 📝 Lessons Learned

- **Tool granularity matters.** Splitting a single RAG chain into five purpose-built tools with distinct prompt templates dramatically improved response quality. A summary tool shouldn't use the same prompt as a detailed specs tool.
- **Image handling in Streamlit is tricky.** Storing image URLs in LangChain message objects (`additional_kwargs`) doesn't survive Streamlit reruns. The solution was a dedicated `st.session_state` dictionary keyed by message index, keeping image data completely separate from message content.
- **Post-retrieval filtering complements vector search.** Vector similarity alone can't handle structured constraints like "under $5,000." Combining semantic retrieval with metadata filtering (price, bike type) gives users the precision they expect.

---
