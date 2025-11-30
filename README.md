![Cellami Icon](assets/Cellami.png)

# Cellami: Local AI for Excel

**Cellami** is a private, local AI assistant that lives inside your Excel. It lets you chat with your data and documents without ever sending them to the cloud.

---

## 🚀 Features

*   **🔒 Private & Local:** Runs on Ollama. Your data stays on your Mac.
*   **💬 Chat with Data:** Select cells and ask questions.
*   **📚 Knowledge Base:** Upload PDFs and Word docs to let the AI answer questions based on your files.
*   **📊 Table Analysis:** Deep insights into your structured data.

---

## 📦 Installation (For Users - Mac Only)

### 1. Install the App
1.  Download or copy the **`Cellami.app`** file to your computer (e.g., in your Applications folder).
2.  Double-click **`Cellami.app`** to start it. (wait 10-15 seconds for it to appear)
    *   **Note:** You won't see a window pop up! Look for the **Cellami icon** in your Mac's top menu bar (System Tray).

### 2. Connect Excel
1.  Open **Microsoft Excel**.
2.  You need to "sideload" the add-in manifest once:
    *   Go to this folder on your Mac:
        `~/Library/Containers/com.microsoft.Excel/Data/Documents/wef` (if the "wef" folder doesn't exist, you will need to create it)
        *(Press `Cmd+Shift+G` in Finder and paste that path)*.
    *   Copy the `manifest.prod.xml` file into that folder.
3.  Restart Excel.
4.  Go to the **Insert** tab > and click the drop down arrow next to **My Add-ins**. 
5.  Select **Cellami** from the Developer add-ins section.

---

## 💻 Development (For Contributors)

### Prerequisites
*   **Node.js** (for Frontend)
*   **Python 3.12+** (for Backend)

### Setup
1.  **Frontend:**
    ```bash
    cd frontend
    npm install
    ```
2.  **Backend:**
    ```bash
    pip install -r requirements.txt
    ```

### Running Locally
1.  **Start Frontend:**
    ```bash
    cd frontend
    npm run dev
    ```
2.  **Start Backend:**
    ```bash
    python main.py
    ```

---

## 🏗️ Building (Distribution)

To create the standalone application:

### 🍎 macOS
**Note:** This is already prebuilt for Mac users within the "Releases" section of this repository.

Run the build script:
```bash
./build_app.sh
```
This will create `dist/Cellami.app`.

### 🪟 Windows
**Note:** You must run this on a Windows machine to create a Windows executable.

Run the batch script:
```cmd
build_app.bat
```
This will create `dist/Cellami.exe`.
