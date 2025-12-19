# Data Privacy Policy

**Effective Date:** December 19, 2025

## 1. Local-First Architecture
Cellami is designed with a "Privacy by Design" and "Local-First" philosophy. This means that:
*   **Storage:** All documents, embeddings, and chat history are stored exclusively on your local machine (specifically in `~/.cellami`).
*   **Processing:** All AI processing is performed deeply locally using your own hardware (via Ollama).
*   **No Cloud Transmission:** No data is transmitted to Cellami servers or third-party cloud LLM providers (like OpenAI or Anthropic) unless you explicitly configure an external provider in the future.

## 2. Data Retention
*   **User Control:** You retain full ownership of your data.
*   **Deletion:** You can delete documents from the Knowledge Base at any time via the application interface. Deleting a document removes its text content, chunks, and vector embeddings from your local disk.
*   **Logs:** Application logs are stored in `~/.cellami/cellami.log` for troubleshooting purposes.
    *   *Redaction:* Sensitive content (such as the full text of user queries or document chunks) is redacted from logs to prevent accidental leakage.
    *   *Retention:* Logs are not automatically uploaded anywhere.

## 3. Data Sensitivity
*   **Encryption:** Application-level encryption is not currently implemented. All data at rest (JSON files, Markdown) is stored in standard plaintext formats on your local disk. We **strictly rely** on your operating system's full-disk encryption (e.g., FileVault on macOS, BitLocker on Windows) to protect this data in the event of device theft.
*   **Access Control:** Access to the application's API is protected by a session token generated at runtime, ensuring that only the authorized frontend application can access your data.

## 4. Updates
This policy reflects the current state of the application. Future updates that introduce cloud features or telemetry will require an update to this policy and explicit user consent.
