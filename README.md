# AI Portfolio Projects 🤖

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-Latest-green.svg)](https://langchain-ai.github.io/langgraph/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4-orange.svg)](https://openai.com)
[![SQLite](https://img.shields.io/badge/SQLite-3.0+-lightgrey.svg)](https://sqlite.org)

Welcome to my AI portfolio repository! This is a collection of hands-on AI projects designed to showcase practical applications of artificial intelligence and machine learning technologies. As a learning journey, each project explores different aspects of AI development, from natural language processing to intelligent assistants.

## 🎯 Purpose

This repository serves as both a learning playground and a portfolio showcase. Each project demonstrates:
- Real-world AI applications
- Modern AI development practices
- Integration of multiple AI technologies
- Problem-solving through intelligent systems

## 🚀 Projects

### 1. Cannondale Bikes AI Assistant
**[📁 View Project](./src/cannondale_bikes_assistant/)**

An intelligent assistant that helps users find the perfect Cannondale bike based on their needs and preferences. This project demonstrates:
- Retrieval-Augmented Generation (RAG) for accurate product recommendations
- Interactive web interface for seamless user experience
- Vector database integration for efficient information retrieval
- Conversational AI powered by OpenAI's GPT models

**Key Features:**
- Natural language bike recommendations
- Product specification lookup
- Interactive chat interface
- Persistent conversation history

## 🛠 Technologies Used

- **Python 3.9+**: Core programming language
- **LangGraph**: Framework for building stateful AI applications
- **Streamlit**: Web application framework for rapid prototyping
- **OpenAI GPT-4**: Large language model for natural language understanding
- **ChromaDB**: Vector database for semantic search
- **SQLite**: Lightweight database for data persistence
- **LangChain**: Framework for developing LLM applications

## 🐍 Environment Setup

This project uses Conda for environment management to ensure consistent dependencies across different systems.

### Prerequisites
- [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed
- Python 3.9 or higher

### Installation Steps

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd ai_portfolio_projects
   ```

2. **Create a new Conda environment:**
   ```bash
   conda create -n ai_portfolio python=3.9
   ```

3. **Activate the environment:**
   ```bash
   conda activate ai_portfolio
   ```

4. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Set up environment variables:**
   Create a `.env` file in the root directory and add your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

### Running Projects

Each project includes its own README with specific running instructions. Generally:

```bash
# Navigate to a project directory
cd src/cannondale_bikes_assistant/

# Run the application (example for Streamlit apps)
streamlit run app/app.py
```

## 📖 Learning Resources

This repository demonstrates concepts from:
- Large Language Models (LLMs)
- Retrieval-Augmented Generation (RAG)
- Vector Databases and Semantic Search
- Conversational AI
- Web Application Development

## 🔄 Continuous Learning

This repository is actively maintained and expanded as I explore new AI technologies and frameworks. Each project represents a step in my AI learning journey, with code that evolves and improves over time.

## 📝 Notes

- All projects are designed for educational purposes
- Code includes extensive comments for learning reference
- Each project can be run independently
- Environment isolation ensures clean dependency management

## 📄 License

This project is licensed under the MIT License - see below for details.

### MIT License

Copyright (c) 2025 [Your Name]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

*This repository represents my ongoing exploration of AI technologies and serves as a practical portfolio of AI applications.*
