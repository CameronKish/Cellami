# Cellami - Microsoft AppSource Store Listing

This document contains all the text content needed for your Microsoft Partner Center submission.
Copy each section into the corresponding field in Partner Center.

---

## BASIC INFORMATION

### App Name
```
Cellami
```

### Short Description (Summary)
*Max 100 characters. Shown in search results.*
```
Private, local AI assistant for Excel. Chat with your data without sending it to the cloud.
```

### Keywords (for search)
```
AI, artificial intelligence, local AI, private AI, Excel assistant, data analysis, document QA, RAG, LLM, Ollama, chat with data, spreadsheet AI, knowledge base, offline AI
```

---

## DETAILED DESCRIPTION

*Max 4,000 characters. This is your main sales pitch.*

```
Cellami brings the power of AI directly to Excel — without ever sending your data to the cloud.

**🔒 Private by Design**
Unlike cloud-based AI tools, Cellami runs entirely on your computer. Your spreadsheets, documents, and conversations stay on your machine. No API keys to manage, no data privacy concerns, no subscription fees.

**💬 Chat With Your Data**
Select cells, tables, or entire ranges and ask questions in plain English. Cellami understands context and provides intelligent insights about your data — from quick summaries to complex analysis.

**📚 Knowledge Base**
Upload PDFs, Word documents, and other files to create a personal knowledge base. Ask questions and get answers grounded in your own documents, with full source citations.

**🔍 Audit Trail**
Every AI response includes transparency. See exactly which document passages were used to generate each answer. Trace back, verify, and trust the results.

**⚡ Key Features**
• Query individual cells or entire tables
• Chat naturally with follow-up questions
• Upload documents for context-aware answers
• View source citations for all responses
• Works with any Ollama-compatible AI model
• Supports multiple models (Llama, Mistral, Gemma, and more)

**🛠 How It Works**
Cellami uses a lightweight companion app that runs in your system tray. The app connects to Ollama (a free, open-source AI runtime) to power the AI capabilities. Simply install both, and the Excel add-in handles the rest.

**👥 Perfect For**
• Financial analysts who can't share sensitive data
• Researchers working with confidential documents  
• Anyone who wants AI assistance without cloud dependency
• Teams that need full control over their data

**💻 Requirements**
• Windows or macOS
• Microsoft Excel (Desktop or Online)
• Ollama (free download from ollama.com)
• 8GB RAM minimum (16GB recommended)

Get started in minutes. No account required. No credit card. Just install and go.
```

---

## WHAT'S NEW (Release Notes)
*For version updates*

```
Version 1.2.0
• Improved connection stability
• Better error messages with troubleshooting guidance
• Enhanced browser compatibility instructions for Chrome/Edge
• Performance optimizations for large documents
```

---

## SUPPORT INFORMATION

### Support URL
```
https://cellami.vercel.app/support.html
```

### Privacy Policy URL
```
https://cellami.vercel.app/privacy.html
```

### Terms of Use / License URL
```
https://cellami.vercel.app/terms.html
```

### Publisher Website
```
https://github.com/CameronKish/Cellami
```

---

## IMAGES CHECKLIST

### Required Icons (Already Created)
- [x] 16x16 icon (`frontend/public/assets/icon-16.png`)
- [x] 32x32 icon (`frontend/public/assets/icon-32.png`)
- [x] 64x64 icon (`frontend/public/assets/icon-64.png`)
- [x] 80x80 icon (`frontend/public/assets/icon-80.png`)

### Screenshots Needed
*Recommended: 1280x720 or 1366x768 resolution*

You need to capture these screenshots:

1. **Chat Feature** - Show the add-in panel open with a conversation
2. **Cell Query** - Show cells selected with AI response
3. **Table Analysis** - Show table selection with analysis results
4. **Knowledge Base** - Show documents uploaded and a query
5. **Audit Trail** - Show source citations for an answer

**Tips:**
- Use real, realistic data (not "test" or "lorem ipsum")
- Don't show empty states or loading screens
- Crop out personal information
- Light mode preferred by Microsoft

---

## CATEGORIES

### Primary Category
```
Productivity
```

### Secondary Category (if available)
```
Data & Analytics
```

### Products
```
Excel
```

---

## PRICING

### License Type
```
Free
```

### Notes
```
The add-in is free. Users need to install the free companion desktop app and Ollama (also free) to use it.
```

---

## LANGUAGE & MARKETS

### Default Language
```
English (United States)
```

### Markets
```
All markets (or select based on your preference)
```

---

## ADDITIONAL NOTES FOR CERTIFICATION

*Paste into the "Notes for Certification" field in Partner Center. Also provide the PDF guide.*

```
TESTING REQUIREMENTS
====================
This add-in requires a companion desktop application to function.

1. Download Cellami desktop app from: https://github.com/CameronKish/Cellami/releases
2. Download Ollama from: https://ollama.com
3. Run: ollama pull gemma3:4b
4. Launch Cellami desktop app (runs in system tray/menu bar)
5. Test the add-in in Excel Online (excel.office.com) for easiest setup

NO ACCOUNTS OR LICENSE KEYS REQUIRED
All processing happens locally on the test machine.

See attached certification_guide.pdf for detailed testing instructions.
```

---

## SUBMISSION CHECKLIST

Before submitting, verify:

- [ ] Manifest XML is valid (no schema errors)
- [ ] All icon sizes are present and accessible
- [ ] Privacy, Terms, and Support pages are live on Vercel
- [ ] Desktop app download links work
- [ ] At least 3 screenshots captured
- [ ] Certification guide PDF is ready to upload
- [ ] All URLs in manifest match live deployed URLs
