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

### 4. Generate the README

- Use the analyzed codebase to fill in the README template
- Infer as much as possible from the code:
  - Project name from directory name or config
  - Tech stack from imports and dependencies
  - How it works from the code flow
  - Environment variables from `os.getenv()` calls or `.env` references
- **Ask the user** for information that cannot be inferred:
  - Problem statement / motivation
  - Target audience
  - Any specific features to highlight

### 5. Write the README

- Create `README.md` at the project root:
```
  {project_root}/README.md
```
- Confirm successful creation:
```
  ✅ README.md created at {project_root}/README.md
```

---

## File Path Resolution Rules

| App File Location | Project Root | README Location |
|-------------------|--------------|-----------------|
| `src/project_name/app/app.py` | `src/project_name/` | `src/project_name/README.md` |
| `src/project_name/main.py` | `src/project_name/` | `src/project_name/README.md` |
| `projects/my_app/src/app.py` | `projects/my_app/` | `projects/my_app/README.md` |

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
<img src="screenshots/app_main.png" alt="Application Screenshot" width="800">
</div>

---

## 📋 Project Overview

{2-3 paragraphs explaining:}
- The real-world problem this project addresses
- How this solution approaches the problem
- What makes this implementation interesting or unique

### 🎯 Key Concepts Explored

{List the AI/ML skills demonstrated with brief explanations}

- **{Concept 1}**: {One sentence explanation of how it's applied}
- **{Concept 2}**: {One sentence explanation of how it's applied}
- **{Concept 3}**: {One sentence explanation of how it's applied}

---

## 🏗️ How It Works

{Explain the system architecture in plain language. Avoid jargon—write as if explaining to a curious colleague.}

<div align="center">
<img src="screenshots/architecture.png" alt="System Architecture" width="800">
</div>

### The Pipeline

{Step-by-step breakdown of how data/queries flow through the system}

1. **{Step Name}**: {What happens and why}
2. **{Step Name}**: {What happens and why}
3. **{Step Name}**: {What happens and why}

### {Feature Section Title}

{Explain a key feature with supporting screenshot}

<div align="center">
<img src="screenshots/feature_example.png" alt="Feature Screenshot" width="800">
</div>

{Add more feature sections as needed}

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

{Show 3-5 example queries or use cases}

**{Use Case Category 1}:**
- "{example query or action}"
- "{example query or action}"

**{Use Case Category 2}:**
- "{example query or action}"
- "{example query or action}"

---

## 📁 Project Structure

```
{repo-name}/
├── src/
│   ├── {main_module}/
│   │   ├── app.py
│   │   └── ...
├── data/
├── screenshots/
├── .env.example
├── pyproject.toml
└── README.md
```

---

## 🔮 Future Improvements

- [ ] {Potential enhancement 1}
- [ ] {Potential enhancement 2}
- [ ] {Potential enhancement 3}

---

## 📝 Lessons Learned

{Optional section: 2-3 key insights gained from building this project}

---
```

---

## Writing Guidelines

### Tone
- **Informational**: Focus on what the project does and how
- **Educational**: Explain concepts clearly for readers learning from your work
- **Accessible**: Avoid unnecessary jargon; define technical terms when first used
- **Confident**: Present your work professionally without over-qualifying

### Screenshots (no more than 3)
- Use placeholder paths: `screenshots/{descriptive_name}.png`. Be sure to describe the write screenshot to take.
- Center images with `<div align="center">` wrapper
- Set consistent width (typically 800px)
- Include meaningful alt text

### Code Blocks (no more than 3)
- Use appropriate language tags for syntax highlighting
- Keep examples concise and runnable
- Include comments for non-obvious lines

### Section Priority
1. **Overview**: Hook the reader—what problem, what solution
2. **How It Works**: Show technical depth and understanding
3. **Tech Stack**: Quick reference for technologies used
4. **Getting Started**: Make it easy to run locally
5. **Examples**: Demonstrate practical usage

---

## Example Prompts to Use This Skill

- "Create a README for my LangChain RAG chatbot project"
- "Help me write documentation for my fine-tuned sentiment classifier"
- "Generate a portfolio README for my AI agent that searches documents"