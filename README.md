![Cellami Icon](assets/Cellami.png)

# Cellami: Local AI for Excel

**Cellami** is a private, local AI assistant that lives inside your Excel. It lets you chat with your data and documents without ever sending them to the cloud.

---

## 🚀 Features

### 1. Knowledge Base
Upload your own documents (PDFs, Word, etc.) and ask questions based on them. Cellami uses your local files to provide accurate, context-aware answers.
<video src="assets/Cellami%20Knowledge%20Base%20Demo.mp4" controls="controls" style="max-width: 100%;"></video>

### 2. Query Cells
Select any cell in your spreadsheet and ask Cellami to process it. Perfect for quick analysis, summarization, or extraction tasks on specific data points. Activate the knowledge base for context-aware answers.
<video src="assets/Cellami%20Cell%20Query%20Demo.mp4" controls="controls" style="max-width: 100%;"></video>

### 3. Audit Answers
Trace back every answer to its source. The Audit tab shows you exactly which document chunks were used to generate a response, ensuring transparency and trust.
<video src="assets/Cellami%20Audit%20Demo.mp4" controls="controls" style="max-width: 100%;"></video>

### 4. Chat with Tables
Have a free-form conversation with your data. The Chat tab allows for open-ended questions and follow-ups, maintaining context throughout your session.
<video src="assets/Cellami%20Chat%20Demo.mp4" controls="controls" style="max-width: 100%;"></video>

### 5. Query Tables
Analyze entire tables at once. Select a range of data, and Cellami will analyze the full table or individual entries sequentially to provide comprehensive insights. Activate the knowledge base for context-aware answers.
<video src="assets/Cellami%20Table%20Query%20Demo.mp4" controls="controls" style="max-width: 100%;"></video>

---

## 📦 App Download & Installation

### 1. Install the App (Mac Only)
1.  Download the **`Cellami_Mac.zip`** file from the **Releases** section on GitHub.
2.  Unzip the file. You will find **`Cellami.app`** and **`manifest.prod.xml`**.
3.  Move **`Cellami.app`** to your Applications folder.
4.  Double-click **`Cellami.app`** to start it. (wait ~15 seconds for it to appear)
    *   **Note:** You won't see a window pop up! Look for the **Cellami icon** in your Mac's top menu bar (System Tray).

### 2. Connect Excel

#### 🍎 macOS
1.  Open **Microsoft Excel**.
2.  You need to "sideload" the add-in manifest once:
    *   Go to this folder on your Mac:
        `~/Library/Containers/com.microsoft.Excel/Data/Documents/wef` (if the "wef" folder doesn't exist, you will need to create it)
        *(Press `Cmd+Shift+G` in Finder and paste that path)*.
    *   Copy the **`manifest.prod.xml`** file (from the unzipped folder) into that folder.
3.  Restart Excel.
4.  Go to the **Insert** tab > and click the drop down arrow next to **My Add-ins**. 
5.  Select **Cellami** from the Developer add-ins section.

#### 🪟 Windows
1.  Create a folder on your computer (e.g., `C:\Manifests`) and put `manifest.prod.xml` inside it.
2.  **Share the folder:**
    *   Right-click the folder > **Properties** > **Sharing** > **Share...**
    *   Add yourself (or "Everyone") and click **Share**.
    *   Note the "Network Path" (e.g., `\\Your-PC\Manifests`).
3.  **Trust the folder in Excel:**
    *   Open Excel > **File** > **Options**.
    *   Go to **Trust Center** > **Trust Center Settings...**
    *   Select **Trusted Add-in Catalogs**.
    *   In "Catalog Url", paste the **Network Path** you copied.
    *   Click **Add Catalog**.
    *   Check the box **"Show in Menu"**.
    *   Click **OK** twice to restart Excel.
4.  **Load the Add-in:**
    *   **Close the Options window** to return to your spreadsheet.
    *   Go to the **Insert** tab (at the top of the screen, next to Home).
    *   Click **My Add-ins** (or "Add-ins" > "My Add-ins").
    *   Click **SHARED FOLDER** at the top of the popup window.
    *   Select **Cellami** and click **Add**.

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

### 1. Install Dependencies
Run this in your terminal (same for Mac/Windows):

```bash
# Frontend
cd frontend
npm install
cd ..

# Backend
pip install -r requirements.txt
```

### 2. Run Build Script
*   **🍎 macOS (zsh):**
    ```zsh
    ./build_app.sh
    ```
    *Creates `dist/Cellami.app`*

*   **🪟 Windows (PowerShell):**
    ```powershell
    .\build_app.bat
    ```
    *Creates `dist/Cellami.exe`*
