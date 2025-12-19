# Security Policy

## Supported Versions
Only the latest version of Cellami is currently supported with security updates.

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability
If you discover a security vulnerability in this project, please report it privately to the maintainer. **Do not create a public GitHub issue.**

## Security Controls
We implement the following controls to protect the application:

### 1. Authentication
*   **Session Tokens:** The backend API is protected by a cryptographically strong session token (`secrets.token_hex`) generated at runtime.
*   **Token Injection:** In production, this token is securely injected directly into the application HTML, preventing exposure via unauthenticated API endpoints.
*   **No Default Passwords:** There are no default passwords or hardcoded credentials.

### 2. Secrets Management
*   **Ephemeral Secrets:** Session tokens exist only in memory.
*   **Ephemeral Secrets:** Session tokens exist only in memory.
*   **Source Control:** No secrets are checked into the source code repository.

### 3. Network Security
*   **Localhost Binding:** The server binds strictly to `127.0.0.1` to prevent network exposure.
*   **API Restriction:** The unauthenticated token endpoint is disabled in production to preventing local port scanning attacks.

### 4. Input Validation
*   **Sanitization:** File uploads undergo strict filename sanitization to prevent directory traversal attacks.
*   **Typosquatting:** We pin exact versions of all dependencies in `requirements.txt` to prevent dependency confusion attacks.

### 5. Logging & Privacy
*   **Redaction:** Sensitive user data input is redacted from application logs ("Privacy by Design").
