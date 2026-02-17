# AI Portfolio README Generator

## Skill Purpose
Generate professional README.md files for AI/ML portfolio projects that showcase technical skills, problem-solving ability, and intellectual curiosity.

---

## Arguments
- `$ARGUMENTS`: The relative file path to the app. For example `src/01_cannondale_bikes_assistant/app/app2.py`

## Instructions

When asked to create a README for an AI portfolio project, gather the following information and generate a well-structured document.

## Execution Steps

### 1. Parse the File Path Argument

- Accept the app file path as the primary argument
- Validate the file exists and is a Python file (`.py`)
- Determine the project root: navigate **up from the app file** until you find the project boundary
  - Project root is typically 2 levels up from an `app/` directory
  - Example: `src/01_cannondale_bikes_assistant/app/app2.py` → project root is `src/01_cannondale_bikes_assistant/`

### 2. Read and Analyze the Codebase

- **Read the main app file** specified in the argument
- **Follow all imports** to build a complete picture:
  - Identify `import` and `from ... import` statements
  - Resolve relative imports (e.g., `from ..utils import helper`)
  - Resolve local/project imports (e.g., `from src.module import func`)
  - Skip standard library and external package imports (e.g., `import os`, `from langchain import ...`)
- **Read each imported local file** recursively
- **Build a mental model** of:
  - Main application entry point and flow
  - Key classes and functions
  - External services/APIs used
  - Data models and schemas
  - Configuration patterns

### 3. Check for Existing README

Before creating the README, check if `README.md` already exists at the project root:
```
{project_root}/README.md
```

- **If README.md exists**: Ask the user:
```
  A README.md already exists at {project_root}/README.md
  Do you want to overwrite it? (yes/no)
```
  - If **no**: Stop execution and inform the user
  - If **yes**: Proceed with generation

### 4. Run the App and Capture Screenshots

#### 4.1 Create Screenshots Directory
```bash
mkdir -p {project_root}/png
```

#### 4.2 Start the Application
- Identify the app type (Streamlit, Gradio, FastAPI, etc.) from imports
- Run the appropriate command:
  - **Streamlit**: `streamlit run {app_file_path}`
  - **Gradio**: `python {app_file_path}`
  - **Other**: Infer from code structure

#### 4.3 Capture Demonstration Screenshots
Using browser control (`claude --chrome`) or Playwright MCP:

1. **Main Interface Screenshot** (`png/app_main.png`)
   - Navigate to the app URL (e.g., `http://localhost:8501`)
   - Capture the initial/home state of the application

2. **Functionality Screenshots** (2-3 examples)
   - Analyze the app to identify key use cases
   - For each screenshot:
     - Formulate a realistic user query/action based on the app's purpose
     - Execute the query in the app
     - Wait for the response to fully render
     - Capture screenshot with descriptive name: `png/example_{description}.png`
   - Examples:
     - `png/example_search_results.png` - Shows search functionality
     - `png/example_chat_response.png` - Shows AI response
     - `png/example_filter_results.png` - Shows filtering capability

#### 4.4 Screenshot Requirements
- **Quantity**: 2-3 screenshots demonstrating different features
- **Quality**: Ensure full responses are visible before capture
- **Naming**: Use lowercase with underscores: `png/{descriptive_name}.png`

#### 4.5 Stop the Application
- Terminate the running app process after screenshots are captured

### 5. Extract Key Code Snippets

Identify and extract 2-3 code snippets that showcase technical implementation:

#### What to Extract
- **Core logic**: Main algorithm, pipeline, or processing function
- **Tool/Agent definitions**: Custom tools, agent configurations
- **Data processing**: Key transformations or retrieval logic

#### Extraction Rules
- Keep snippets focused: 15-40 lines each
- Include necessary context (imports, class definition if method)
- Add comments to clarify non-obvious logic
- Format for README inclusion with proper syntax highlighting

#### Example Snippet Categories
```yaml
snippets:
  - name: "RAG Retrieval Pipeline"
    file: "src/retriever.py"
    function: "retrieve_documents"
    lines: "45-78"

  - name: "Custom Tool Definition"
    file: "src/tools.py"
    function: "search_bikes"
    lines: "12-55"

  - name: "Agent Configuration"
    file: "src/agent.py"
    section: "AGENT_PROMPT and executor setup"
    lines: "100-135"
```

### 6. Generate the README

- Use the analyzed codebase to fill in the README template
- Infer as much as possible from the code:
  - Project name from directory name or config
  - Tech stack from imports and dependencies
  - How it works from the code flow
  - Environment variables from `os.getenv()` calls or `.env` references
- Reference captured screenshots with correct paths (`png/{name}.png`)
- Include extracted code snippets in appropriate sections
- **Ask the user** for information that cannot be inferred:
  - Problem statement / motivation
  - Target audience
  - Any specific features to highlight

### 7. Write the README

- Create `README.md` at the project root:
```
  {project_root}/README.md
```
- Confirm successful creation:
```
  ✅ README.md created at {project_root}/README.md
  ✅ Screenshots saved to {project_root}/png/
     - app_main.png
     - example_{name1}.png
     - example_{name2}.png
```

---

## File Path Resolution Rules

| App File Location | Project Root | README Location | Screenshots |
|-------------------|--------------|-----------------|-------------|
| `src/project_name/app/app.py` | `src/project_name/` | `src/project_name/README.md` | `src/project_name/png/` |
| `src/project_name/main.py` | `src/project_name/` | `src/project_name/README.md` | `src/project_name/png/` |
| `projects/my_app/src/app.py` | `projects/my_app/` | `projects/my_app/README.md` | `projects/my_app/png/` |

**Heuristics for finding project root:**
1. Look for `pyproject.toml`, `setup.py`, `requirements.txt`, or `.env` files
2. If app is in an `app/` or `src/` subdirectory, go up one level
3. Stop at the directory containing project config files

### Required Information to Collect

```yaml
project:
  name: ""                    # Project title
  emoji: ""                   # Single relevant emoji for the title
  tagline: ""                 # Brief subtitle (e.g., "A RAG System with Memory")

problem:
  what: ""                    # What problem does this solve?
  who: ""                     # Who experiences this problem?
  why_it_matters: ""          # Why is solving this valuable?

skills_demonstrated:
  - ""                        # List of AI/ML concepts explored
  # Examples: RAG Architecture, Fine-tuning, Vector Search,
  # Prompt Engineering, Agent Design, Tool Use, etc.

technology:
  llm: ""                     # LLM used (GPT-4, Claude, Llama, etc.)
  embeddings: ""              # Embedding model (if applicable)
  vector_store: ""            # Vector database (if applicable)
  framework: ""               # Main framework (LangChain, LlamaIndex, etc.)
  frontend: ""                # UI framework (Streamlit, Gradio, etc.)
  other: []                   # Additional tools/libraries

screenshots:
  main: "png/app_main.png"
  examples:
    - path: "png/example_{name}.png"
      query: ""               # The query/action performed
      description: ""         # What this demonstrates

code_snippets:
  - name: ""                  # Descriptive name for the snippet
    description: ""           # What this code does
    code: ""                  # The actual code (15-40 lines)

setup:
  python_version: ""          # Minimum Python version
  package_manager: ""         # pip, poetry, uv, conda
  env_vars: []                # Required environment variables
  prerequisites: []           # Other requirements (API keys, accounts)
```

---

## README Template

Generate the README using this structure:

```markdown
# {emoji} {Project Name}
***{Tagline}***

<div align="center">
<img src="png/app_main.png" alt="Application Screenshot" width="800">
</div>

---

## 📋 Project Overview

{2-3 paragraphs explaining:}
- The hypothetical real-world problem this project addresses
- How this solution approaches the hypothetical problem
- What makes this implementation interesting or unique

### 🎯 Key Concepts Explored

{List the AI/ML skills demonstrated with brief explanations}

- **{Concept 1}**: {One sentence explanation of how it's applied}
- **{Concept 2}**: {One sentence explanation of how it's applied}
- **{Concept 3}**: {One sentence explanation of how it's applied}

---

## 🏗️ How It Works

{Explain the system architecture in plain language. Avoid jargon—write as if explaining to a curious colleague.}
Create a markdown visual mock-up of the architecture (prompt, tools, agents), etc showing how it all comes together.

### The Pipeline

{Step-by-step breakdown of how data/queries flow through the system}

1. **{Step Name}**: {What happens and why}
2. **{Step Name}**: {What happens and why}
3. **{Step Name}**: {What happens and why}

### {Feature Section Title}

{Explain a key feature with a code snippet}

```python
{extracted_code_snippet_1}
```

### {Feature Section Title}

{Explain another key feature}

```python
{extracted_code_snippet_2}
```

{Add more feature sections as needed, limit to 2-3 code blocks total}

---

## 🛠️ Tech Stack

```
🧠 LLM:         {model name and version}
🔍 Embeddings:  {embedding model}
🗄️ Data Store:  {database or vector store}
🌐 Frontend:    {UI framework}
🔗 Framework:   {AI/ML framework}
📦 Other:       {additional notable tools}
```

---

## 🚀 Getting Started

### Prerequisites

- Python {version}+
- {Package manager}
- {Required API keys or accounts}

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/{username}/{repo-name}.git
cd {repo-name}
```

2. **Install dependencies**
```bash
{installation command}
```

3. **Set up environment variables**

Create a `.env` file in the project root:
```bash
{ENV_VAR_1}=your_value_here
{ENV_VAR_2}=your_value_here
```

4. **{Any additional setup steps}**
```bash
{commands}
```

### Running the Application

```bash
{run command}
```

The app will be available at `http://localhost:{port}`

---

## 💡 Example Usage

{Show 2-3 example interactions with screenshots}

### {Example 1 Title}

**Query:** "{the actual query used}"

<div align="center">
<img src="png/example_{name1}.png" alt="{description}" width="800">
</div>

{Brief explanation of what the app returned and why it's useful}

### {Example 2 Title}

**Query:** "{the actual query used}"

<div align="center">
<img src="png/example_{name2}.png" alt="{description}" width="800">
</div>

{Brief explanation of what the app returned and why it's useful}

---

## 📁 Project Structure

```
{repo-name}/
├── app/
│   └── app.py
├── src/
│   └── {modules}/
├── data/
├── png/
│   ├── app_main.png
│   ├── example_{name1}.png
│   └── example_{name2}.png
├── .env.example
├── pyproject.toml
└── README.md
```

---
```

---

## Writing Guidelines

### Tone
- **Informational**: Focus on what the project does and how
- **Educational**: Explain concepts clearly for readers learning from your work
- **Accessible**: Avoid unnecessary jargon; define technical terms when first used
- **Confident**: Present your work professionally without over-qualifying

### Screenshots
- **Location**: All screenshots saved to `png/` folder in project root
- **Quantity**: Maximum 3 screenshots total
- **Naming**: Lowercase with underscores: `app_main.png`, `example_search.png`
- Center images with `<div align="center">` wrapper
- Set consistent width (typically 800px)
- Include meaningful alt text

### Code Blocks
- **Quantity**: Maximum 2-3 code snippets in README
- **Length**: 15-40 lines each, focused on key logic
- **Selection Priority**:
  1. Core algorithm or pipeline logic
  2. Tool/agent definitions
  3. Interesting data transformations
- Use appropriate language tags for syntax highlighting
- Include brief comments for non-obvious lines

### Section Priority
1. **Overview**: Hook the reader—what problem, what solution
2. **How It Works**: Show technical depth with code snippets
3. **Tech Stack**: Quick reference for technologies used
4. **Getting Started**: Make it easy to run locally
5. **Examples**: Demonstrate practical usage with screenshots

---

## Screenshot Capture Workflow

### For Streamlit Apps
```bash
# 1. Start the app
streamlit run {app_file_path} &

# 2. Wait for startup
sleep 5

# 3. Use Playwright MCP or browser control to:
#    - Navigate to http://localhost:8501
#    - Capture main interface
#    - Enter test queries
#    - Capture results

# 4. Kill the app
pkill -f streamlit
```

### Recommended Test Queries
Analyze the app's purpose and generate realistic queries:

| App Type | Example Queries |
|----------|-----------------|
| Product Search | "Show me bikes under $5000", "Compare mountain vs road bikes" |
| Document Q&A | "What are the main findings?", "Summarize chapter 3" |
| Code Assistant | "Explain this function", "Find bugs in this code" |
| Data Analysis | "Show trends for Q4", "Compare regions" |

---

## Example Prompts to Use This Skill

- "Create a README for my LangChain RAG chatbot project"
- "Help me write documentation for my fine-tuned sentiment classifier"
- "Generate a portfolio README for my AI agent that searches documents"