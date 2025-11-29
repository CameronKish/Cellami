# Cellami: Local AI for Excel

**Cellami** is a powerful, privacy-first Excel Add-in that brings the power of Large Language Models (LLMs) directly into your spreadsheet. It runs entirely on your local machine using **Ollama**, ensuring your data never leaves your computer.

![Cellami Icon](assets/Cellami.png)

## 🚀 Features

*   **Local & Private:** Powered by Ollama. No data is sent to the cloud.
*   **Chat with Data:** Select cells and ask questions about them.
*   **RAG (Retrieval Augmented Generation):** Build a Knowledge Base from your own documents (PDF, DOCX, TXT, etc.) and query them inside Excel.
*   **Structured Table Analysis:** Analyze user-defined tables with context from your local Knowledge Base, enabling deep insights into your structured data.

## 🛠️ Prerequisites

Before you begin, ensure you have the following installed:

1.  **[Ollama](https://ollama.com/)**: The engine for running local models.
    *   After installing, run: `ollama pull mistral:7b` (or your preferred model).
2.  **[Python 3.12+](https://www.python.org/)**: For the backend server.
3.  **[Node.js](https://nodejs.org/)**: For the frontend build tools.

## 📦 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/cameronkish/cellami.git
cd cellami
```

### 2. Backend Setup (Python)
It is recommended to use a virtual environment.

```bash
# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Frontend Setup (Node.js)
```bash
cd frontend
npm install
cd ..
```

## 🏃‍♂️ Running the App (Development)

To run the app in development mode (with hot-reloading):

1.  **Start the Backend:**
    ```bash
    # In the root directory
    uvicorn main:app --reload --port 8000
    ```

2.  **Start the Frontend:**
    ```bash
    # In a new terminal, inside frontend/
    cd frontend
    npm run dev
    ```

3.  **Sideload into Excel:**
    *   **Mac:** Copy `manifest.xml` to `~/Library/Containers/com.microsoft.Excel/Data/Documents/wef`.
    *   **Windows:** Add the folder containing `manifest.xml` to **Excel > Options > Trust Center > Trusted Add-in Catalogs**.
    *   Restart Excel and look for the **Cellami** tab.

## 🚢 Deployment (Bundled App)

For end-users, you can package Cellami into a single executable file (`.exe` or `.app`) so they don't need to install Python or Node.js.

See **[deployment_options.md](deployment_options.md)** for the full guide on how to build and distribute the bundled app.

## 🧠 Knowledge Base

Cellami supports RAG (Retrieval Augmented Generation).
1.  Go to the **Docs** tab in the add-in.
2.  Upload your documents (PDF, Markdown, Text).
3.  Cellami will chunk and embed them locally.
4.  In the **Query** or **Chat** tab, enable "Use RAG" to let the AI answer questions based on your documents.

## 📄 License

[MIT License](LICENSE)
