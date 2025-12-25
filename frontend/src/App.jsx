import React, { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import './index.css';
import './design-system.css';

import rehypeSanitize from 'rehype-sanitize';
/* global Excel, Office */

// In production (hosted PWA), we must point to the local loopback backend.
// In development (Vite), we use the proxy set in vite.config.js.
const API_BASE = import.meta.env.PROD ? "https://127.0.0.1:8000/api" : "/api";

let authToken = "";

const fetchAPI = async (url, options = {}) => {
  if (!authToken) {
    console.warn(`Blocked API call to ${url} because authToken is missing.`);
    // Return a fake 401 response so the caller handles it, but backend is never hit.
    return { ok: false, status: 401, json: async () => ({ error: "No client token" }) };
  }
  const headers = { ...options.headers };
  headers['X-API-Token'] = authToken;

  // Add PNA fetch option to explicitly mark this as a local network request
  // This tells Chrome the request is intentionally targeting a private network address
  return fetch(url, {
    ...options,
    headers,
    // Chrome's Private Network Access feature requires this for localhost requests
    targetAddressSpace: 'local'
  });
};


// --- Reusable UI Components ---

const Card = ({ children, className = "" }) => (
  <div className={`bg-white rounded-xl shadow-sm border border-slate-200 p-5 ${className}`}>
    {children}
  </div>
);

const Button = ({ children, onClick, variant = "primary", className = "", disabled = false, ...props }) => {
  const baseStyle = "px-4 py-2 rounded-lg font-medium transition-all duration-200 flex items-center justify-center gap-2 active:scale-95";
  const variants = {
    primary: "bg-indigo-600 text-white hover:bg-indigo-700 shadow-md hover:shadow-lg disabled:bg-indigo-300",
    secondary: "bg-white text-slate-700 border border-slate-200 hover:bg-slate-50 hover:border-slate-300 shadow-sm",
    danger: "bg-red-50 text-red-600 hover:bg-red-100 border border-red-100",
    ghost: "text-slate-500 hover:text-indigo-600 hover:bg-indigo-50"
  };

  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={`${baseStyle} ${variants[variant]} ${disabled ? 'cursor-not-allowed opacity-70' : ''} ${className}`}
      {...props}
    >
      {children}
    </button>
  );
};

const Input = ({ className = "", ...props }) => (
  <input
    className={`w-full max-w-full px-3 py-2 bg-white border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500/20 focus:border-indigo-500 transition-all box-border ${className}`}
    {...props}
  />
);

const Badge = ({ children, variant = "neutral" }) => {
  const variants = {
    neutral: "bg-slate-100 text-slate-600",
    success: "bg-emerald-100 text-emerald-700",
    warning: "bg-amber-100 text-amber-700",
    error: "bg-red-100 text-red-700",
    primary: "bg-indigo-100 text-indigo-700"
  };
  return (
    <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${variants[variant]}`}>
      {children}
    </span>
  );
};

// --- Main App Component ---

import MarkdownViewer from './components/MarkdownViewer';
import Markdown from 'react-markdown';
import rehypeRaw from 'rehype-raw';
import remarkGfm from 'remark-gfm';
import DownloadPage from './components/DownloadPage';

// ... (existing imports)

const App = () => {
  const [activeTab, setActiveTab] = useState('query');
  const [settings, setSettings] = useState({ config: {}, prompts: [] });
  const [documents, setDocuments] = useState([]);
  const [excludedDocs, setExcludedDocs] = useState([]); // Track deselected documents
  const [isOfficeInitialized, setIsOfficeInitialized] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false); // Global processing state
  const [isAuthReady, setIsAuthReady] = useState(false); // Wait for auth token before rendering

  // Viewer State
  const [viewerOpen, setViewerOpen] = useState(false);
  const [viewerContent, setViewerContent] = useState('');
  const [viewerHighlight, setViewerHighlight] = useState('');
  const [viewerFilename, setViewerFilename] = useState('');

  // About Modal State
  const [showAbout, setShowAbout] = useState(false);

  // Close source viewer when switching tabs
  useEffect(() => {
    setViewerOpen(false);
  }, [activeTab]);

  // Load persisted state for formatting options
  const loadPersistedState = () => {
    try {
      const saved = localStorage.getItem('queryViewState');
      return saved ? JSON.parse(saved) : {};
    } catch {
      return {};
    }
  };
  const persisted = loadPersistedState();

  const [includeColors, setIncludeColors] = useState(persisted.includeColors !== false);
  const [includeStyles, setIncludeStyles] = useState(persisted.includeStyles !== false);
  const [debugPrompts, setDebugPrompts] = useState(persisted.debugPrompts || false);
  const [showInspectButton, setShowInspectButton] = useState(false);
  const [developerMode, setDeveloperMode] = useState(false);

  const fetchSettings = useCallback(async () => {
    try {
      const res = await fetchAPI(`${API_BASE}/settings`);
      if (res.ok) {
        const data = await res.json();
        setSettings(data);
      }
    } catch (e) {
      console.error("Failed to fetch settings", e);
    }
  }, []);

  const refreshDocs = useCallback(async () => {
    try {
      const res = await fetchAPI(`${API_BASE}/list-documents`);
      if (res.ok) {
        const data = await res.json();
        setDocuments(data.documents);
      }
    } catch (e) {
      console.error("Failed to fetch docs", e);
    }
  }, []);

  // Helper: Fetch a new token (used for init and auto-recovery)
  const fetchToken = useCallback(async () => {
    try {
      // 1. Check for Injected Token (Production)
      if (window.__CELLAMI_TOKEN__) {
        console.log("Auth: Found injected token.");
        return window.__CELLAMI_TOKEN__;
      }

      // 2. Fetch from backend
      // Add cache: no-store AND timestamp to force-bust any aggressive SW/Browser caching
      // Include targetAddressSpace for Chrome's Private Network Access compliance
      const res = await fetch(`${API_BASE}/auth/token?t=${Date.now()}`, {
        cache: "no-store",
        targetAddressSpace: 'local'
      });
      if (res.ok) {
        const data = await res.json();
        console.log("Auth: Token fetch success.");
        return data.token;
      } else {
        const msg = `Auth Check Failed: ${res.status} ${res.statusText}`;
        console.warn(msg);
        setConnectionErrorMessage(msg);
      }
    } catch (e) {
      console.warn("Auth: Connector logic skipped/failed:", e);
      setConnectionErrorMessage(`Auth Fetch Error: ${e.message}`);
    }
    return null;
  }, []);

  const [connectionError, setConnectionError] = useState(false);
  const [connectionErrorMessage, setConnectionErrorMessage] = useState("");

  useEffect(() => {
    const initAuth = async () => {
      try {
        const token = await fetchToken();

        // 3. If we received a token, verify connectivity/health
        if (token) {
          authToken = token; // Set global
          try {
            // Use settings fetch as a health check
            const res = await fetchAPI(`${API_BASE}/settings`);
            if (res.ok) {
              const data = await res.json();
              setSettings(data);
              refreshDocs(); // Fire and forget
              setConnectionError(false);
            } else {
              const msg = `Init Settings Failed: ${res.status} ${res.statusText}`;
              console.warn(msg);
              setConnectionErrorMessage(msg);
              // Strictly fail if we can't get settings (e.g. 401 from stale token, or backend error)
              setConnectionError(true);
            }
          } catch (e) {
            console.error("Health check failed:", e);
            setConnectionErrorMessage(`Health Check Error: ${e.message}`);
            setConnectionError(true);
          }
        } else {
          // No token found (Backend likely down)
          // Strictly fail connection to prevent empty UI
          console.warn("Init: No token found. Backend is down.");
          setConnectionError(true);
        }
      } catch (e) {
        console.error("Failed to initialize auth", e);
        setConnectionError(true);
      } finally {
        setIsAuthReady(true);
      }
    };

    Office.onReady((info) => {
      if (info.host === Office.HostType.Excel) {
        setIsOfficeInitialized(true);
      }
    });

    initAuth();
  }, [fetchToken]);

  // --- Heartbeat: Monitor Backend Connection ---
  useEffect(() => {
    // Only start monitoring after initial auth attempt is complete
    if (!isAuthReady) return;

    const checkHealth = async () => {
      try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 2000);

        // SMART HEARTBEAT:
        // If we don't have a token, pinging /settings will just cause a 401 log spam.
        // Instead, we should try to GET a token (Token Polling).
        if (!authToken) {
          // console.log("Heartbeat: No token, attempting to acquire...");
          const newToken = await fetchToken();
          if (newToken) {
            // console.log("Heartbeat: Acquired token!");
            authToken = newToken;
            setConnectionError(false); // We are back online!
            fetchSettings();
            refreshDocs();
          } else {
            // Still no token (backend likely down or still starting)
            if (!connectionError) setConnectionError(true);
          }
          return;
        }

        // If we DO have a token, verify it's still good by checking settings
        const res = await fetchAPI(`${API_BASE}/settings`, { signal: controller.signal });
        clearTimeout(timeoutId);

        if (res.ok) {
          if (connectionError) {
            setConnectionError(false);
            fetchSettings();
            refreshDocs();
          }
        } else {
          // 4xx/5xx means server IS reachable but unhappy.
          if (res.status === 401) {
            // Token expired or invalid (e.g. backend restarted).
            // Attempt to Auto-Heal by fetching a new token!
            // console.log("Heartbeat: 401 Unauthorized. Attempting to re-auth...");
            const newToken = await fetchToken();
            if (newToken) {
              // console.log("Heartbeat: Re-auth successful!");
              authToken = newToken;
              setConnectionError(false);
              fetchSettings();
              refreshDocs();
            } else {
              // console.warn("Heartbeat: Re-auth failed.");
              if (!connectionError) setConnectionError(true);
            }
          }
          // Other errors (500, 404) we might ignore for heartbeat purposes
        }
      } catch (e) {
        // Network Error or Timeout => Backend is likely down
        console.warn("Heartbeat missed:", e);
        if (!connectionError) {
          setConnectionErrorMessage(`Heartbeat Lost: ${e.message}`);
          setConnectionError(true);
        }
      }
    };

    // Check every 3 seconds for responsive feedback
    const interval = setInterval(checkHealth, 3000);
    return () => clearInterval(interval);
  }, [isAuthReady, connectionError, fetchToken, fetchSettings, refreshDocs]);

  const handleViewSource = async (filename, chunkText, sourceId = null, directContent = null) => {
    if (directContent) {
      setViewerContent(directContent);
      setViewerFilename(filename);
      setViewerHighlight(chunkText);
      setViewerOpen(true);
      return;
    }
    try {
      // If we have a source ID, try to fetch the full text for better highlighting
      if (sourceId) {
        try {
          const chunkRes = await fetchAPI(`${API_BASE}/chunk/${sourceId}`);
          if (chunkRes.ok) {
            const chunkData = await chunkRes.json();
            if (chunkData.text) {
              chunkText = chunkData.text;
            }
          }
        } catch (e) {
          console.warn("Failed to fetch full chunk text, falling back to snippet", e);
        }
      }

      const res = await fetchAPI(`${API_BASE}/document-content?filename=${encodeURIComponent(filename)}`);
      if (!res.ok) {
        if (res.status === 404) {
          alert("Source content not found. Please re-upload this document to enable the viewer.");
          return;
        }
        throw new Error("Failed to fetch document content");
      }
      const data = await res.json();
      setViewerContent(data.content);
      setViewerFilename(filename);
      setViewerHighlight(chunkText);
      setViewerOpen(true);
    } catch (e) {
      console.error("Failed to open viewer", e);
      alert(`Error: ${e.message}`);
    }
  };

  // Calculate active documents (those not excluded)
  const activeDocs = useMemo(() => {
    return documents.filter(doc => !excludedDocs.includes(doc));
  }, [documents, excludedDocs]);

  const renderContent = () => {
    switch (activeTab) {
      case 'chat':
        return <ChatView settings={settings} handleViewSource={handleViewSource} activeDocs={activeDocs} onProcessingChange={setIsProcessing} />;
      case 'query':
        return <QueryView
          settings={settings}
          refreshSettings={fetchSettings}
          includeColors={includeColors}
          setIncludeColors={setIncludeColors}
          includeStyles={includeStyles}
          setIncludeStyles={setIncludeStyles}
          debugPrompts={debugPrompts}
          setDebugPrompts={setDebugPrompts}
          activeDocs={activeDocs} // Pass active docs for RAG filtering
          onProcessingChange={setIsProcessing}
        />;
      case 'docs':
        return <KnowledgeView
          documents={documents}
          refreshDocs={refreshDocs}
          excludedDocs={excludedDocs}
          setExcludedDocs={setExcludedDocs}
          showInspectButton={showInspectButton}
          onProcessingChange={setIsProcessing}
        />;
      case 'audit':
        return <AuditView onViewSource={handleViewSource} />;
      case 'settings':
        return <SettingsView
          settings={settings}
          refreshSettings={fetchSettings}
          includeColors={includeColors}
          setIncludeColors={setIncludeColors}
          includeStyles={includeStyles}
          setIncludeStyles={setIncludeStyles}
          debugPrompts={debugPrompts}
          setDebugPrompts={setDebugPrompts}
          showInspectButton={showInspectButton}
          setShowInspectButton={setShowInspectButton}
          developerMode={developerMode}
          setDeveloperMode={setDeveloperMode}
          onProcessingChange={setIsProcessing}
        />;
      default:
        return <ChatView settings={settings} handleViewSource={handleViewSource} onProcessingChange={setIsProcessing} />;
    }
  };

  // --- Sub Components ---

  const ConnectionError = ({ message }) => (
    <div className="flex flex-col min-h-screen items-center justify-center bg-slate-50 p-6 text-center">
      {/* Brand Logo with Glow Effect */}
      <div className="mb-8 relative">
        <div className="absolute inset-0 bg-sky-400/20 blur-[50px] rounded-full"></div>
        <img
          src="/Cellami_Template.png"
          alt="Cellami Logo"
          className="relative w-24 h-24 object-contain drop-shadow-xl"
        />
      </div>

      <h2 className="text-2xl font-bold text-slate-900 mb-3 tracking-tight">Connection Failed</h2>

      <p className="text-slate-600 max-w-sm mb-4 leading-relaxed font-medium">
        Cellami is not reachable.<br />
        Please ensure the Cellami app is running.
      </p>

      <button
        onClick={() => window.location.reload()}
        className="min-w-[200px] px-8 py-4 rounded-full bg-sky-600 hover:bg-sky-500 text-white font-bold text-lg shadow-lg shadow-sky-200 transition-all transform hover:-translate-y-1 active:scale-95"
      >
        Retry Connection
      </button>

      {/* Download Redirect for New Users */}
      <div className="mt-8 pt-8 border-t border-slate-200 w-full max-w-xs flex flex-col items-center animate-fade-in-up delay-200">
        <p className="text-slate-500 text-sm mb-3 font-medium">Don't have the companion app?</p>
        <a
          href="https://cellami.ai"
          target="_blank"
          rel="noopener noreferrer"
          className="text-indigo-600 hover:text-indigo-700 text-sm font-bold flex items-center gap-1.5 hover:underline decoration-2 underline-offset-2 transition-colors"
        >
          Download for Mac / Windows
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
          </svg>
        </a>
      </div>

      {/* Subtle Error Message (Bottom Left) */}
      {message && (
        <div className="fixed bottom-2 left-2 max-w-[60%] text-left">
          <pre className="text-red-400/80 text-[10px] font-mono break-words whitespace-pre-wrap">
            {message}
          </pre>
        </div>
      )}

      {/* Build Timestamp (Bottom Right) */}
      <div className="fixed bottom-2 right-2 text-[10px] text-slate-400 font-mono opacity-50">
        Build: {__BUILD_TIMESTAMP__}
      </div>

    </div>
  );

  if (!isOfficeInitialized) {
    return <DownloadPage />;
  }

  if (connectionError) {
    return <ConnectionError message={connectionErrorMessage} />;
  }

  if (!isAuthReady) {
    return (
      <div className="flex h-screen items-center justify-center bg-slate-50">
        <div className="text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-600 mx-auto mb-4"></div>
          <p className="text-slate-600">Connecting to Cellami...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="h-screen app-glass-bg font-sans overflow-hidden selection:bg-indigo-100 selection:text-indigo-900">
      {/* Fixed Header / Navigation */}
      <header className="glass-header">
        <div style={{
          width: '100%',
          height: '100%',
          padding: '0 52px 0 16px',
          borderTop: isProcessing ? '4px solid var(--color-primary)' : 'none',
          transition: 'border-top 0.3s ease',
          display: 'flex',
          alignItems: 'center',
          gap: '12px'
        }}>
          {/* App Logo */}
          <div
            className="flex-shrink-0 cursor-pointer group"
            onClick={() => setShowAbout(true)}
            title="About Cellami"
          >
            <img
              src="/Cellami_Template.png"
              alt="Cellami"
              className="w-8 h-8 object-contain opacity-60 group-hover:opacity-100 group-hover:scale-110 group-hover:drop-shadow-md transition-all duration-300 ease-out"
            />
          </div>

          <nav className="tabs" style={{ marginBottom: 0, flex: 1, overflowX: 'auto', scrollbarWidth: 'none' }}>
            {['query', 'chat', 'docs', 'audit', 'settings'].map((tab) => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                disabled={isProcessing}
                className={`tab ${activeTab === tab ? 'active' : ''}`}
                style={{ textTransform: 'capitalize' }}
                title={isProcessing ? "Please wait for the current process to finish" : ""}
              >
                {tab}
              </button>
            ))}
          </nav>
        </div>
      </header>

      {/* Main Content Area - Overlays to allow scrolling behind header */}
      <main className="glass-content-wrapper">
        <div className="h-full w-full mx-auto">
          {renderContent()}
        </div>
      </main>

      <MarkdownViewer
        isOpen={viewerOpen}
        onClose={() => setViewerOpen(false)}
        content={viewerContent}
        highlightText={viewerHighlight}
        filename={viewerFilename}
      />

      {/* About Modal */}
      {showAbout && (
        <div
          className="fixed inset-0 z-[3000] flex items-center justify-center bg-white/70 backdrop-blur-md animate-in fade-in duration-300"
          onClick={() => setShowAbout(false)}
        >
          <div
            className="glass-panel p-8 max-w-[320px] w-full relative overflow-hidden text-center animate-in zoom-in-95 duration-200"
            onClick={e => e.stopPropagation()}
            style={{
              border: '1px solid rgba(255,255,255,0.5)',
              boxShadow: '0 20px 40px rgba(0,0,0,0.2)',
              borderRadius: '24px'
            }}
          >
            {/* Background Glow */}
            <div className="absolute -top-24 -left-24 w-48 h-48 bg-sky-400/10 blur-[60px] rounded-full"></div>
            <div className="absolute -bottom-24 -right-24 w-48 h-48 bg-indigo-400/10 blur-[60px] rounded-full"></div>

            <div className="flex justify-center mb-6 relative">
              <div className="absolute inset-0 bg-sky-400/20 blur-[50px] rounded-full"></div>
              <img src="/Cellami_Template.png" alt="Cellami Logo" className="w-16 h-16 drop-shadow-lg relative" />
            </div>

            <h2 className="text-2xl font-bold text-slate-800 mb-1" style={{ letterSpacing: '-0.02em' }}>Cellami</h2>
            <p className="text-sky-600 font-mono text-xs mb-6 font-bold tracking-wider">VERSION {typeof __APP_VERSION__ !== 'undefined' ? __APP_VERSION__ : '1.2.0'}</p>

            <p className="text-slate-600 text-sm leading-relaxed mb-8">
              The intelligent companion for Excel.<br />
              Local. Private. Secure.
            </p>

            <div className="flex flex-col gap-3">
              <button onClick={() => setShowAbout(false)} className="btn btn-primary w-full py-2.5" style={{ borderRadius: '12px' }}>Close</button>
              <p className="text-slate-400 text-[9px] uppercase tracking-[0.2em] mt-2 font-bold opacity-80">Empowering Data Analysts</p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

// --- VIEWS ---

const QueryView = ({
  settings,
  refreshSettings,
  includeColors,
  includeStyles,
  debugPrompts,
  activeDocs = [], // Default to empty array if not provided
  onProcessingChange
}) => {
  // Load persisted state from localStorage
  const loadPersistedState = () => {
    try {
      const saved = localStorage.getItem('queryViewState');
      return saved ? JSON.parse(saved) : {};
    } catch {
      return {};
    }
  };

  const persisted = loadPersistedState();

  const [prompt, setPrompt] = useState(persisted.prompt || '');
  const [selectedPromptId, setSelectedPromptId] = useState(persisted.selectedPromptId || '');
  const [useRag, setUseRag] = useState(persisted.useRag || false);
  const [sourceOnly, setSourceOnly] = useState(persisted.sourceOnly || false);
  const [queryMode, setQueryMode] = useState(persisted.queryMode || 'cell'); // 'cell' or 'table'
  const [headerAddress, setHeaderAddress] = useState(persisted.headerAddress || '');
  const [contentAddress, setContentAddress] = useState(persisted.contentAddress || '');
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState('');
  const [updateSuccess, setUpdateSuccess] = useState(false);
  // const [debugStatus, setDebugStatus] = useState('');
  const [, setDebugStatus] = useState('');
  const [outputLocation, setOutputLocation] = useState(persisted.outputLocation || 'right');
  const [isSaving, setIsSaving] = useState(false);
  const [saveName, setSaveName] = useState('');
  const [isDeleting, setIsDeleting] = useState(false);
  const [showOverwriteConfirm, setShowOverwriteConfirm] = useState(false);
  const [pendingBatch, setPendingBatch] = useState(null);
  const [currentTaskId, setCurrentTaskId] = useState(null);
  const abortControllerRef = useRef(null);

  // Smart Query Refinement State
  const [showRefinementModal, setShowRefinementModal] = useState(false);
  const [proposedQueries, setProposedQueries] = useState([]); // Changed from single string to array
  const [pendingTableBatch, setPendingTableBatch] = useState(null);
  const [refinementStrategy, setRefinementStrategy] = useState(persisted.refinementStrategy || 'review'); // 'none', 'auto', 'review'

  // Iterative Processing State
  // const [iterationCount, setIterationCount] = useState(1);
  const [analyzeSequentially, setAnalyzeSequentially] = useState(false);
  const [sequenceMode, setSequenceMode] = useState('row'); // 'row' or 'col'
  const [smartDetectionMsg, setSmartDetectionMsg] = useState(null);
  const [alignmentStatus, setAlignmentStatus] = useState(null); // 'aligned-row', 'aligned-col', 'misaligned', null
  const [elapsedSeconds, setElapsedSeconds] = useState(0);

  // Timer Logic
  useEffect(() => {
    let interval;
    if (loading) {
      setElapsedSeconds(0);
      interval = setInterval(() => {
        setElapsedSeconds(prev => prev + 1);
      }, 1000);
    } else {
      clearInterval(interval);
    }
    return () => clearInterval(interval);
  }, [loading]);

  // Sync loading state with parent
  useEffect(() => {
    if (onProcessingChange) {
      onProcessingChange(loading);
    }
  }, [loading, onProcessingChange]);

  // Save state to localStorage whenever it changes
  useEffect(() => {
    localStorage.setItem('queryViewState', JSON.stringify({
      prompt,
      selectedPromptId,
      useRag,
      sourceOnly,
      queryMode,
      headerAddress,
      contentAddress,
      outputLocation,
      includeColors,
      includeStyles,
      debugPrompts,
      refinementStrategy
    }));
  }, [prompt, selectedPromptId, useRag, sourceOnly, queryMode, headerAddress, contentAddress, outputLocation, includeColors, includeStyles, debugPrompts, refinementStrategy]);

  const initiateSave = () => {
    if (!prompt) {
      alert("Please enter a prompt first.");
      return;
    }
    setIsSaving(true);
    setSaveName('');
    setDebugStatus('');
  };

  const cancelSave = () => {
    setIsSaving(false);
    setSaveName('');
    setDebugStatus('');
  };

  const confirmSave = async () => {
    setDebugStatus('Starting save...');
    if (!saveName) {
      setDebugStatus('Error: No name entered');
      alert("Please enter a name for the prompt.");
      return;
    }

    try {
      setDebugStatus('Preparing data...');
      const currentPrompts = settings?.prompts || [];
      const updatedPrompts = [...currentPrompts, { id: Date.now(), text: prompt, name: saveName }];
      const updatedSettings = { ...settings, prompts: updatedPrompts };

      setDebugStatus('Sending request...');
      const res = await fetchAPI(`${API_BASE}/settings`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(updatedSettings)
      });

      setDebugStatus(`Response: ${res.status}`);

      if (res.ok) {
        setDebugStatus('Success!');
        if (refreshSettings) await refreshSettings();
        setIsSaving(false);
        setSaveName('');
      } else {
        const errText = await res.text();
        console.error("Save failed:", errText);
        setDebugStatus(`Error: ${res.status} ${errText}`);
        alert(`Failed to save prompt: ${res.status} ${errText}`);
      }
    } catch (e) {
      console.error("Save exception:", e);
      setDebugStatus(`Exception: ${e.message}`);
      alert(`Failed to save prompt: ${e.message}`);
    }
  };

  const updatePrompt = async () => {
    if (!selectedPromptId || !prompt) return;

    try {
      const currentPrompts = settings?.prompts || [];
      const updatedPrompts = currentPrompts.map(p =>
        p.id.toString() === selectedPromptId ? { ...p, text: prompt } : p
      );
      const updatedSettings = { ...settings, prompts: updatedPrompts };

      const res = await fetchAPI(`${API_BASE}/settings`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(updatedSettings)
      });

      if (res.ok) {
        if (refreshSettings) await refreshSettings();
        setUpdateSuccess(true);
        setTimeout(() => setUpdateSuccess(false), 2000);
      } else {
        alert("Failed to update prompt.");
      }
    } catch (e) {
      console.error("Update failed:", e);
      alert("Failed to update prompt: " + e.message);
    }
  };

  const initiateDelete = () => {
    if (!selectedPromptId) return;
    setIsDeleting(true);
  };

  const cancelDelete = () => {
    setIsDeleting(false);
  };

  const confirmDelete = async () => {
    const updatedPrompts = settings.prompts.filter(p => p.id.toString() !== selectedPromptId);
    const updatedSettings = { ...settings, prompts: updatedPrompts };

    try {
      const res = await fetchAPI(`${API_BASE}/settings`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(updatedSettings)
      });
      if (res.ok) {
        if (refreshSettings) await refreshSettings();
        setSelectedPromptId('');
        setPrompt('');
        setIsDeleting(false);
      }
    } catch (e) {
      console.warn("Deleted prompt failure", e);
      alert("Failed to delete prompt");
    }
  };

  const handlePromptSelect = (e) => {
    const id = e.target.value;
    setSelectedPromptId(id);
    if (id && settings) {
      const p = settings.prompts.find(p => p.id.toString() === id);
      if (p) setPrompt(p.text);
    } else {
      setPrompt('');
    }
  };

  // Helper: Convert 6-char hex to 3-char hex
  const to3CharHex = (hex6) => {
    if (!hex6 || hex6.length !== 7) return null;
    const r = Math.round(parseInt(hex6.substr(1, 2), 16) / 17);
    const g = Math.round(parseInt(hex6.substr(3, 2), 16) / 17);
    const b = Math.round(parseInt(hex6.substr(5, 2), 16) / 17);
    return `#${r.toString(16)}${g.toString(16)}${b.toString(16)}`.toUpperCase();
  };

  // Helper: Extract formatted cell values from one or more ranges
  const extractFormattedCells = async (rangesInput, context, options = {}) => {
    const ranges = Array.isArray(rangesInput) ? rangesInput : [rangesInput];

    // 1. Load properties for ALL ranges
    for (const r of ranges) {
      r.load("values, rowCount, columnCount");
    }
    await context.sync();

    const allRows = [];

    // 2. Process each range
    for (const r of ranges) {
      const cellData = [];
      // Load all cell properties first
      for (let row = 0; row < r.rowCount; row++) {
        cellData[row] = [];
        for (let col = 0; col < r.columnCount; col++) {
          const cell = r.getCell(row, col);
          cell.load("numberFormat, text");

          if (options.includeColors) {
            cell.format.fill.load("color");
          }

          const fontProps = [];
          if (options.includeColors) fontProps.push("color");
          if (options.includeStyles) fontProps.push("bold", "italic");

          if (fontProps.length > 0) {
            cell.format.font.load(fontProps.join(", "));
          }

          cellData[row][col] = cell;
        }
      }
      await context.sync();

      // Build rows for this range
      for (let row = 0; row < r.rowCount; row++) {
        const rowData = [];
        for (let col = 0; col < r.columnCount; col++) {
          const cell = cellData[row][col];
          // Ensure value is a string (handle cases where Excel returns an object)
          let value = String(cell.text);

          // Fix for narrow columns displaying "#####"
          // Check if the text is composed entirely of hashes and whitespace
          const isAllHashes = value &&
            value.includes('#') &&
            value.replace(/[#\s]/g, '').length === 0;

          if (isAllHashes) {
            // Fallback to raw value
            if (r.values && r.values[row] && r.values[row][col] !== undefined) {
              const rawValue = r.values[row][col];
              value = String(rawValue);
            }
          }

          // Apply markdown for bold/italic
          if (options.includeStyles) {
            if (cell.format.font.bold) value = `**${value}**`;
            if (cell.format.font.italic) value = `*${value}*`;
          }

          if (options.includeColors) {
            // Add background color if not default
            const bgColor = cell.format.fill.color;
            if (bgColor && bgColor !== '#FFFFFF' && bgColor !== 'white') {
              const hex3 = to3CharHex(bgColor);
              if (hex3) value += ` [bg:${hex3}]`;
            }

            // Add font color if not default
            const fgColor = cell.format.font.color;
            if (fgColor && fgColor !== '#000000' && fgColor !== 'black') {
              const hex3 = to3CharHex(fgColor);
              if (hex3) value += ` [fg:${hex3}]`;
            }
          }

          rowData.push(value || '');
        }
        allRows.push(rowData);
      }
    }
    return allRows;
  };

  // Helper: Extract formatted table as markdown from one or more ranges
  const extractFormattedTable = async (rangesInput, context, options = {}) => {
    const ranges = Array.isArray(rangesInput) ? rangesInput : [rangesInput];
    const hasExplicitHeader = ranges.length > 1;

    // Load layout info to determine merge direction
    if (hasExplicitHeader) {
      ranges[0].load("rowIndex");
      ranges[1].load("rowIndex");
      await context.sync();
    }

    let headerRows = [];
    let contentRows = [];

    if (hasExplicitHeader) {
      // Extract separately to handle merging
      headerRows = await extractFormattedCells(ranges[0], context, options);
      contentRows = await extractFormattedCells(ranges[1], context, options);

      // Strip formatting from header rows
      headerRows = headerRows.map(row => row.map(cell => {
        let clean = String(cell || "");
        // Remove color tags
        clean = clean.replace(/ \[bg:#[0-9a-fA-F]{3}\]/g, "").replace(/ \[fg:#[0-9a-fA-F]{3}\]/g, "");
        // Remove bold/italic wrappers
        clean = clean.replace(/^[*\s]+|[*\s]+$/g, "");
        return clean;
      }));
    } else {
      contentRows = await extractFormattedCells(ranges[0], context, options);
    }

    let finalRows = [];

    if (hasExplicitHeader) {
      // Determine direction: If start rows match, it's horizontal (side-by-side)
      const isHorizontal = ranges[0].rowIndex === ranges[1].rowIndex;

      if (isHorizontal) {
        // Merge row by row (Horizontal)
        const rowCount = Math.max(headerRows.length, contentRows.length);
        for (let i = 0; i < rowCount; i++) {
          const h = headerRows[i] || Array(headerRows[0]?.length || 0).fill("");
          const c = contentRows[i] || Array(contentRows[0]?.length || 0).fill("");
          finalRows.push([...h, ...c]);
        }
      } else {
        // Stack (Vertical)
        finalRows = [...headerRows, ...contentRows];
      }
    } else {
      finalRows = contentRows;
    }

    if (finalRows.length === 0) return "";

    // Build markdown table
    let legendParts = [];
    if (options.includeStyles) {
      legendParts.push("**text** = bold, *text* = italic");
    }
    if (options.includeColors) {
      legendParts.push("[bg:#RGB] = background color, [fg:#RGB] = font color. Refer to colors by name (e.g. 'red'), never hex codes");
    }

    let markdown = "";
    if (legendParts.length > 0) {
      markdown += "LEGEND: " + legendParts.join(", ") + ".\n\n";
    }

    if (hasExplicitHeader) {
      // Header row (always the first row of our processed set)
      markdown += "| " + finalRows[0].join(" | ") + " |\n";
      markdown += "|" + finalRows[0].map(() => "---").join("|") + "|\n";

      // Data rows
      for (let i = 1; i < finalRows.length; i++) {
        markdown += "| " + finalRows[i].join(" | ") + " |\n";
      }
    } else {
      // No explicit header: treat all rows as data, no separator line
      for (let i = 0; i < finalRows.length; i++) {
        markdown += "| " + finalRows[i].join(" | ") + " |\n";
      }
    }

    return markdown;
  };

  const handleRefinementOrProcess = async (batchData) => {
    // Smart Query Refinement Logic
    if (queryMode === 'table' && useRag) {
      if (refinementStrategy === 'review') {
        setStatus('Refining queries...');
        abortControllerRef.current = new AbortController();
        try {
          const res = await fetchAPI(`${API_BASE}/refine-query`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              inputs: batchData.inputs,
              prompt: prompt,
              model: settings?.config?.model_name
            }),
            signal: abortControllerRef.current.signal
          });

          if (!res.ok) {
            const errText = await res.text();
            console.error("Refinement start failed:", res.status, errText);
            throw new Error(`Failed to start refinement: ${res.status} ${errText}`);
          }
          const { task_id } = await res.json();
          setCurrentTaskId(task_id); // Track for cancellation

          // Poll for refinement results
          while (true) {
            if (abortControllerRef.current?.signal.aborted) {
              throw new Error("Refinement cancelled");
            }

            const statusRes = await fetchAPI(`${API_BASE}/batch-status/${task_id}`);
            if (!statusRes.ok) throw new Error("Failed to check refinement status");

            const statusData = await statusRes.json();

            if (statusData.status === 'completed') {
              setProposedQueries(statusData.results);
              setPendingTableBatch(batchData);
              setShowRefinementModal(true);
              setStatus('');
              setCurrentTaskId(null);
              return true; // Refinement finished (paused for user review)
            } else if (statusData.status === 'failed') {
              throw new Error(statusData.error || "Refinement failed");
            } else if (statusData.status === 'cancelled') {
              throw new Error("Refinement cancelled");
            } else {
              // Update progress
              if (statusData.message) {
                setStatus(statusData.message);
              }
              await new Promise(r => setTimeout(r, 1000));
            }
          }

        } catch (e) {
          console.error("Refinement failed", e);
          setStatus(`Refinement Error: ${e.message}`);
          setCurrentTaskId(null);
          return false; // Stop here
        }
      } else if (refinementStrategy === 'auto') {
        // Auto mode: just pass the flag to the backend
        // No UI interruption needed
      }
    }

    // Standard Execution (Cell Mode, Table Mode without RAG, or Auto/None Strategy)
    await processBatch(
      batchData.inputs,
      batchData.outputSheetId,
      batchData.outputAddress,
      null, // searchQuery (legacy)
      refinementStrategy === 'review' ? 'provided' : refinementStrategy // Backend expects 'provided' if we send specific queries, but here we are just starting. Actually if 'auto' or 'none', we pass that.
    );
    setStatus('Done!');
    return false; // Indicates finished
  };

  const runBatch = async () => {
    setLoading(true);
    setStatus('Processing...');
    let paused = false;

    try {
      const batchData = await Excel.run(async (context) => {
        let range;
        let inputs;
        let outputRange;

        if (queryMode === 'table') {
          if (!contentAddress) {
            throw new Error("Please set the Content range.");
          }

          // Helper to extract sheet name from address (e.g. "Sheet1!A1" -> "Sheet1")
          const getSheetName = (addr) => {
            if (!addr) return null;
            const parts = addr.split('!');
            if (parts.length > 1) {
              // Handle quoted sheet names like 'My Sheet'!A1
              return parts[0].replace(/'/g, '');
            }
            return "ActiveSheet"; // Fallback if no sheet specified (implies active)
          };

          // Pre-check: Ensure Header and Content are on the same sheet
          if (headerAddress && contentAddress) {
            const hSheet = getSheetName(headerAddress);
            const cSheet = getSheetName(contentAddress);
            if (hSheet !== cSheet && hSheet !== "ActiveSheet" && cSheet !== "ActiveSheet") {
              throw new Error(`Header and Content ranges must be on the same worksheet.\n(Header: ${hSheet}, Content: ${cSheet})`);
            }
          }

          // Smart Detection Logic (Already handled in UI, but double check or just use current state)
          let headerRangeObj = null;
          let contentRangeObj = null;

          try {
            contentRangeObj = context.workbook.worksheets.getActiveWorksheet().getRange(contentAddress);
            if (headerAddress) {
              headerRangeObj = context.workbook.worksheets.getActiveWorksheet().getRange(headerAddress);
            }
          } catch (e) {
            // Wrap generic Excel "invalid argument" error with helpful context
            if (e.message && e.message.includes("The argument is invalid or missing")) {
              throw new Error("Invalid Range Error: The selected ranges could not be found.\n\nPlease ensure:\n1. The ranges exist on the active worksheet.\n2. You haven't deleted the referenced cells.");
            }
            throw e;
          }

          range = contentRangeObj;
          range.load("rowCount, columnCount");
          await context.sync();

          inputs = [];

          if (analyzeSequentially) {
            // Sequential Processing
            if (sequenceMode === 'row') {
              // Row by Row
              for (let r = 0; r < range.rowCount; r++) {
                const currentRow = range.getRow(r);
                // Check for empty row (optional optimization, but let's keep it simple first)
                // We need to load values to check emptiness
                currentRow.load("values");
                await context.sync();

                const rowValues = currentRow.values[0];
                const isEmpty = rowValues.every(c => c === "" || c === null || c === undefined);

                if (!isEmpty) {
                  const rangesToExtract = [];
                  if (headerRangeObj) rangesToExtract.push(headerRangeObj);
                  rangesToExtract.push(currentRow);

                  const tableMarkdown = await extractFormattedTable(rangesToExtract, context, { includeColors, includeStyles });
                  inputs.push(tableMarkdown);
                } else {
                  // Push empty string to maintain index alignment? 
                  // Or just skip? If we skip, output alignment might be tricky if we want 1:1 map.
                  // Let's skip for now as per requirement "skipping rows that are blank"
                  // But wait, if we skip, where do we write the result? 
                  // If we write to the right of the *original* row, we need to track the index.
                  // For simplicity in this version: We will process non-empty rows and write them to a new list.
                  // The output range will be a contiguous block of results.
                  // If the user wants 1:1 alignment with skips, we'd need to insert blanks in results.
                  // Let's assume we skip empty rows and just output results sequentially.
                }
              }

              // Output: Column to the right, height = number of inputs
              if (inputs.length > 0) {
                if (outputLocation === 'right') {
                  outputRange = range.getCell(0, range.columnCount).getResizedRange(inputs.length - 1, 0);
                } else if (outputLocation === 'below') {
                  outputRange = range.getCell(range.rowCount, 0).getResizedRange(inputs.length - 1, 0);
                }
              }

            } else {
              // Column by Column
              for (let c = 0; c < range.columnCount; c++) {
                const currentCol = range.getColumn(c);
                currentCol.load("values");
                await context.sync();

                const colValues = currentCol.values.map(v => v[0]);
                const isEmpty = colValues.every(c => c === "" || c === null || c === undefined);

                if (!isEmpty) {
                  const rangesToExtract = [];
                  if (headerRangeObj) rangesToExtract.push(headerRangeObj);
                  rangesToExtract.push(currentCol);

                  const tableMarkdown = await extractFormattedTable(rangesToExtract, context, { includeColors, includeStyles });
                  inputs.push(tableMarkdown);
                }
              }

              // Output: Rows below, width = number of inputs
              // Actually if we process columns, we probably want results below each column?
              // If "Rows below" is selected, we usually want a row of results.
              if (inputs.length > 0) {
                if (outputLocation === 'below') {
                  // Write as a row below
                  outputRange = range.getCell(range.rowCount, 0).getResizedRange(0, inputs.length - 1);
                } else if (outputLocation === 'right') {
                  // Write as a column to the right
                  outputRange = range.getCell(0, range.columnCount).getResizedRange(inputs.length - 1, 0);
                }
              }
            }

          } else {
            // Single Block Processing (Original Logic but for full range)
            const rangesToExtract = [];
            if (headerRangeObj) rangesToExtract.push(headerRangeObj);
            rangesToExtract.push(range);

            const tableMarkdown = await extractFormattedTable(rangesToExtract, context, { includeColors, includeStyles });
            inputs.push(tableMarkdown);

            // Output: Single cell
            if (outputLocation === 'right') {
              outputRange = range.getCell(0, range.columnCount);
            } else if (outputLocation === 'below') {
              outputRange = range.getCell(range.rowCount, 0);
            }
          }

          if (inputs.length === 0) {
            throw new Error("No valid data found to process.");
          }

          if (outputLocation === 'new-sheet') {
            // outputRange = null; 
          }
        } else {
          // Cell Mode
          range = context.workbook.getSelectedRange();
          range.load("values, rowCount, columnCount");
          await context.sync();

          // Extract formatted cells
          const allFormattedRows = await extractFormattedCells(range, context, { includeColors, includeStyles });

          if (outputLocation === 'below') {
            // For "below" mode: flatten all cells and write results in the same layout
            inputs = allFormattedRows.flat();
            outputRange = range.getOffsetRange(range.rowCount, 0);
            outputRange = outputRange.getResizedRange(range.rowCount - 1, range.columnCount - 1);
          } else if (outputLocation === 'right') {
            // For "right" mode: only use first column, write to column on right
            inputs = allFormattedRows.map(row => row[0]);
            outputRange = range.getOffsetRange(0, 1);
            outputRange = outputRange.getResizedRange(range.rowCount - 1, 0);
          } else {
            // New sheet mode (cell batch)
            inputs = allFormattedRows.flat();
            // Defer creation
          }
        }

        // Sanitize inputs to ensure they are all strings (Pydantic requirement)
        inputs = inputs.map(i => {
          if (i === null || i === undefined) return "";
          return String(i);
        });

        // Debug Prompts Logic
        if (debugPrompts) {
          const debugSheet = context.workbook.worksheets.add("Prompt Debug");
          debugSheet.activate();

          const debugRange = debugSheet.getRange("A1");
          debugRange.values = [["DEBUG: PROMPT & DATA"]];
          debugRange.format.font.bold = true;

          const promptRange = debugSheet.getRange("A3");
          promptRange.values = [["User Prompt:"]];
          promptRange.format.font.bold = true;

          const promptValueRange = debugSheet.getRange("B3");
          promptValueRange.values = [[prompt]];

          const dataHeaderRange = debugSheet.getRange("A5");
          dataHeaderRange.values = [["Generated Inputs (Data sent to AI):"]];
          dataHeaderRange.format.font.bold = true;

          // Write inputs vertically
          if (inputs.length > 0) {
            const dataRange = debugSheet.getRange("A6").getResizedRange(inputs.length - 1, 0);
            // Inputs might be markdown strings or values. Ensure they are strings.
            const inputValues = inputs.map(i => [String(i)]);
            dataRange.values = inputValues;
          }

          await context.sync();
          setLoading(false);
          setStatus('Debug sheet created.');
          return null; // Stop processing
        }

        // Check if output range has content (skip for new sheet)
        if (outputLocation !== 'new-sheet') {
          outputRange.load('values');
          await context.sync();

          const hasContent = outputRange.values.some(row =>
            row.some(cell => cell && cell.toString().trim() !== '')
          );

          if (hasContent) {
            // Store pending data and show confirmation
            outputRange.load("address");
            outputRange.worksheet.load("id");
            await context.sync();

            setPendingBatch({
              inputs,
              outputSheetId: outputRange.worksheet.id,
              outputAddress: outputRange.address
            });
            setShowOverwriteConfirm(true);
            setLoading(false);
            setStatus('');
            return;
          }

          // Load properties for return
          outputRange.load("address");
          outputRange.worksheet.load("id");
          await context.sync();

          return {
            inputs,
            outputSheetId: outputRange.worksheet.id,
            outputAddress: outputRange.address
          };
        } else {
          // New Sheet Mode - Return placeholder
          return {
            inputs,
            outputSheetId: 'NEW_SHEET',
            outputAddress: 'A1'
          };
        }
      });

      if (!batchData) return;

      paused = await handleRefinementOrProcess(batchData);

    } catch (e) {
      if (e.message && (e.message.includes("The argument is invalid or missing") || e.message.includes("Invalid Range Error"))) {
        setStatus("Error: The selected range could not be found. Please ensure you are on the correct sheet.");
      } else {
        setStatus(`Error: ${e.message}`);
      }
    } finally {
      // Only clear loading if we are NOT waiting for refinement modal
      if (!paused) {
        setLoading(false);
      }
    }
  };

  const processBatch = async (inputs, outputSheetId, outputAddress, searchQuery = null, strategy = 'none', searchQueries = null) => {
    // Setup timeout for the entire operation (default 20 mins if not set)
    const timeoutSeconds = settings?.config?.timeout_seconds || 1200;
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeoutSeconds * 1000);

    try {
      // 1. Start the batch job
      const startRes = await fetchAPI(`${API_BASE}/batch-process`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          inputs: inputs,
          prompt: prompt,
          use_rag: useRag,
          source_only: sourceOnly,
          search_query: searchQuery,
          refinement_strategy: strategy,
          search_queries: searchQueries,
          doc_filters: activeDocs // Pass the list of allowed documents
        }),
        signal: controller.signal
      });

      if (!startRes.ok) {
        throw new Error(`Failed to start batch job: ${startRes.status} ${startRes.statusText}`);
      }

      const { task_id } = await startRes.json();
      setCurrentTaskId(task_id);
      setStatus('Processing... (Job started)');

      // 2. Poll for results
      let results = null;
      while (true) {
        if (controller.signal.aborted) {
          // Check if it was a manual cancellation vs timeout
          if (status === 'Cancelled') {
            throw new Error('Operation cancelled by user');
          }
          throw new Error('Operation timed out');
        }

        const statusRes = await fetchAPI(`${API_BASE}/batch-status/${task_id}`, {
          signal: controller.signal
        });

        if (!statusRes.ok) {
          if (statusRes.status === 404) {
            await new Promise(resolve => setTimeout(resolve, 1000));
            continue;
          }
          throw new Error(`Failed to check status: ${statusRes.status}`);
        }

        const statusData = await statusRes.json();

        if (statusData.status === 'completed') {
          results = statusData.results;
          break;
        } else if (statusData.status === 'failed') {
          throw new Error(statusData.error || 'Batch processing failed');
        } else if (statusData.status === 'cancelled') {
          throw new Error('Batch processing cancelled');
        } else {
          // Update status with progress if available
          if (statusData.total && statusData.processed !== undefined) {
            setStatus(`Processing... (${statusData.processed}/${statusData.total})`);
          } else if (statusData.message) {
            setStatus(statusData.message);
          } else {
            setStatus('Processing...');
          }
          // Wait before polling again
          await new Promise(resolve => setTimeout(resolve, 1000));
        }
      }
      clearTimeout(timeoutId);
      setCurrentTaskId(null);

      // 3. Write results to Excel
      setStatus('Writing results...');
      await Excel.run(async (context) => {
        let sheet;
        let range;

        if (outputSheetId === 'NEW_SHEET') {
          sheet = context.workbook.worksheets.add();
          sheet.activate();
          if (results.length > 0) {
            range = sheet.getRange("A1").getResizedRange(results.length - 1, 0);
          } else {
            range = sheet.getRange("A1");
          }
        } else {
          sheet = context.workbook.worksheets.getItem(outputSheetId);
          range = sheet.getRange(outputAddress);
        }

        range.load("rowCount, columnCount");
        await context.sync();

        // Handle structured response - flatten to text first
        const answers = results.map(r => {
          let text = '';
          if (typeof r === 'object' && r !== null && 'answer' in r) {
            text = r.answer || '';
            if (r.sources && r.sources.length > 0) {
              const sourceTags = r.sources.map(s => `${s.id} (${Math.round(s.score * 100)}%)`).join(', ');
              text += ` (Source ID: ${sourceTags})`;
            }
          } else {
            text = String(r);
          }
          // Excel cell limit
          if (text.length > 32000) {
            text = text.substring(0, 32000) + '... [TRUNCATED]';
          }
          return text;
        });

        // Reshape results to match output range dimensions
        const reshapedAnswers = [];
        let idx = 0;
        for (let r = 0; r < range.rowCount; r++) {
          const row = [];
          for (let c = 0; c < range.columnCount; c++) {
            row.push(answers[idx++] || '');
          }
          reshapedAnswers.push(row);
        }

        range.values = reshapedAnswers;
        await context.sync();
      });

    } catch (error) {
      if (error.name === 'AbortError') {
        throw new Error('Request timed out after 20 minutes.');
      }
      if (error instanceof Error && error.message.startsWith('Failed to write')) {
        throw error;
      }
      throw new Error(`Processing failed: ${error.message}`);
    }
  };

  const confirmRefinement = async () => {
    if (!pendingTableBatch) return;

    setShowRefinementModal(false);
    setLoading(true);
    setStatus('Processing with refined queries...');

    try {
      await processBatch(
        pendingTableBatch.inputs,
        pendingTableBatch.outputSheetId,
        pendingTableBatch.outputAddress,
        null, // legacy search_query
        'provided', // strategy is now 'provided' since we are sending the list
        proposedQueries // Pass the list of refined queries
      );
      setStatus('Done!');
    } catch (e) {
      setStatus(`Error: ${e.message}`);
    } finally {
      setLoading(false);
      setPendingTableBatch(null);
    }
  };

  const cancelRefinement = () => {
    setShowRefinementModal(false);
    setPendingTableBatch(null);
    setLoading(false);
    setStatus('');
  };

  const confirmOverwrite = async () => {
    if (!pendingBatch) return;

    setShowOverwriteConfirm(false);
    setLoading(true);
    setStatus('Processing...');

    let paused = false;

    try {
      paused = await handleRefinementOrProcess(pendingBatch);
    } catch (e) {
      setStatus(`Error: ${e.message}`);
    } finally {
      if (!paused) {
        setLoading(false);
      }
      setPendingBatch(null);
    }
  };


  const cancelOverwrite = () => {
    setShowOverwriteConfirm(false);
    setPendingBatch(null);
  };

  const checkAlignment = async (hAddr, cAddr) => {
    if (!hAddr || !cAddr) {
      setAlignmentStatus(null);
      setSmartDetectionMsg(null);
      return null;
    }

    try {
      return await Excel.run(async (context) => {
        const sheet = context.workbook.worksheets.getActiveWorksheet();
        const hRange = sheet.getRange(hAddr);
        const cRange = sheet.getRange(cAddr);
        hRange.load(["rowCount", "columnCount", "rowIndex", "columnIndex"]);
        cRange.load(["rowCount", "columnCount", "rowIndex", "columnIndex"]);
        await context.sync();

        let status = 'misaligned';
        let msg = "Warning: Header and Content ranges do not align by row or column.";

        // 1. Strong Match: Letters (Columns) Match -> Top Headers -> Row Mode
        if (hRange.columnIndex === cRange.columnIndex && hRange.columnCount === cRange.columnCount) {
          status = 'aligned-row';
          msg = "Table ranges align.";
          if (analyzeSequentially && sequenceMode !== 'row') {
            setSequenceMode('row');
            setOutputLocation('right');
          }
        }
        // 2. Strong Match: Numbers (Rows) Match -> Side Headers -> Col Mode
        else if (hRange.rowIndex === cRange.rowIndex && hRange.rowCount === cRange.rowCount) {
          status = 'aligned-col';
          msg = "Table ranges align.";
          if (analyzeSequentially && sequenceMode !== 'col') {
            setSequenceMode('col');
            setOutputLocation('below');
          }
        }
        // 3. Weak Match: Column Counts Match -> Assume Top Headers -> Row Mode
        else if (hRange.columnCount === cRange.columnCount) {
          status = 'aligned-row';
          msg = "Table ranges align.";
          if (analyzeSequentially && sequenceMode !== 'row') {
            setSequenceMode('row');
            setOutputLocation('right');
          }
        }
        // 4. Weak Match: Row Counts Match -> Assume Side Headers -> Col Mode
        else if (hRange.rowCount === cRange.rowCount) {
          status = 'aligned-col';
          msg = "Table ranges align.";
          if (analyzeSequentially && sequenceMode !== 'col') {
            setSequenceMode('col');
            setOutputLocation('below');
          }
        }

        setAlignmentStatus(status);
        setSmartDetectionMsg(msg);
        return status;
      });
    } catch (e) {
      console.log("Invalid range address for check", e);
      setAlignmentStatus(null);
      setSmartDetectionMsg(null);
      return null;
    }
  };

  const setHeaderFromSelection = async () => {
    try {
      await Excel.run(async (context) => {
        const range = context.workbook.getSelectedRange();
        range.load("address");
        await context.sync();
        setHeaderAddress(range.address);
        checkAlignment(range.address, contentAddress);
      });
    } catch (e) {
      console.error(e);
    }
  };

  const setContentFromSelection = async () => {
    try {
      await Excel.run(async (context) => {
        const range = context.workbook.getSelectedRange();
        range.load("address");
        await context.sync();
        setContentAddress(range.address);
        checkAlignment(headerAddress, range.address);
      });
    } catch (e) {
      console.error(e);
    }
  };

  const handleCancel = async () => {
    // Cancel refinement if active (local abort + backend cancel)
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
      setStatus('Cancelling refinement...');

      if (currentTaskId) {
        try {
          await fetchAPI(`${API_BASE}/cancel-batch/${currentTaskId}`, { method: 'POST' });
        } catch (e) { console.error("Failed to cancel backend refinement task", e); }
      }

      setLoading(false);
      setCurrentTaskId(null);
      return;
    }

    if (currentTaskId) {
      try {
        setStatus('Cancelling...');
        await fetchAPI(`${API_BASE}/cancel-batch/${currentTaskId}`, { method: 'POST' });
        // The polling loop will detect the 'cancelled' status or we can manually abort
        // But let's let the polling loop handle the exit to be clean
      } catch (e) {
        console.error("Failed to cancel:", e);
      }
    }
  };

  return (
    <div style={{ height: '100%', overflowY: 'auto', padding: '44px var(--space-xs) var(--space-xs)', background: 'transparent' }}>
      <div style={{ height: '1px', background: 'rgba(255, 255, 255, 0.6)', position: 'fixed', top: '44px', left: 0, right: 0, zIndex: 1001, boxShadow: '0 2px 4px rgba(0,0,0,0.05)' }} />
      <div className="container" style={{ maxWidth: '100%' }}>

        {/* Mode Tabs */}
        <div style={{ display: 'flex', gap: 'var(--space-sm)', marginBottom: 'var(--space-lg)' }}>
          <button
            className={`btn ${queryMode === 'cell' ? 'btn-primary' : 'btn-ghost'}`}
            onClick={() => setQueryMode('cell')}
            style={{ flex: 1, justifyContent: 'center' }}
          >
            Cell Query
          </button>
          <button
            className={`btn ${queryMode === 'table' ? 'btn-primary' : 'btn-ghost'}`}
            onClick={() => setQueryMode('table')}
            style={{ flex: 1, justifyContent: 'center' }}
          >
            Table Analysis
          </button>
        </div>

        {/* Header */}
        <div style={{ marginBottom: 'var(--space-xl)' }}>
          <h1 className="heading-1" style={{ marginBottom: 'var(--space-md)' }}>
            {queryMode === 'table' ? 'Analyze Table' : 'Process Selected Cells'}
          </h1>
          <div className="card" style={{ background: 'var(--color-primary-light)', border: 'none', padding: 'var(--space-md)' }}>
            <p style={{ fontSize: 'var(--font-size-sm)', color: 'var(--color-text-primary)', margin: 0, lineHeight: 1.6 }}>
              {queryMode === 'table'
                ? <span><strong>How it works:</strong> Select a content range (and optional header row). The AI will analyze the data as a single table.</span>
                : <span><strong>How it works:</strong> Select one or more cells in Excel. Each cell will be processed individually with your prompt.</span>
              }
            </p>
          </div>
        </div>

        {/* Table Mode Inputs */}
        {queryMode === 'table' && (
          <div className="card" style={{ marginBottom: 'var(--space-lg)' }}>
            <h2 className="heading-3" style={{ marginBottom: 'var(--space-md)' }}>Table Ranges</h2>

            <div className="input-group">
              <label>Header Range (Optional)</label>
              <div style={{ display: 'flex', gap: 'var(--space-xs)' }}>
                <input
                  className="input"
                  value={headerAddress}
                  onChange={e => {
                    setHeaderAddress(e.target.value);
                    // Optional: debounce checkAlignment(e.target.value, contentAddress)
                  }}
                  onBlur={() => checkAlignment(headerAddress, contentAddress)}
                  placeholder="e.g. A1:D1"
                />
                <button className="btn btn-secondary btn-sm" onClick={setHeaderFromSelection}>Set from Selection</button>
              </div>
            </div>

            <div className="input-group">
              <label>Content Range</label>
              <div style={{ display: 'flex', gap: 'var(--space-xs)' }}>
                <input
                  className="input"
                  value={contentAddress}
                  onChange={e => {
                    setContentAddress(e.target.value);
                    // Optional: debounce checkAlignment(headerAddress, e.target.value)
                  }}
                  onBlur={() => checkAlignment(headerAddress, contentAddress)}
                  placeholder="e.g. A50:D60"
                />
                <button className="btn btn-secondary btn-sm" onClick={setContentFromSelection}>Set from Selection</button>
              </div>
            </div>

            {/* Alignment Indicator */}
            {headerAddress && contentAddress && alignmentStatus && (
              <div style={{
                marginBottom: 'var(--space-md)',
                padding: 'var(--space-sm)',
                borderRadius: 'var(--radius-sm)',
                background: alignmentStatus === 'misaligned' ? 'var(--color-error-bg)' : 'var(--color-primary-light)',
                color: alignmentStatus === 'misaligned' ? 'var(--color-error)' : 'var(--color-primary)',
                fontSize: 'var(--font-size-sm)',
                display: 'flex',
                alignItems: 'center',
                gap: 'var(--space-xs)'
              }}>
                <span style={{ fontSize: 'var(--font-size-xs)', fontWeight: 'var(--font-weight-medium)', color: alignmentStatus === 'misaligned' ? 'var(--color-warning)' : 'var(--color-success)' }}>{alignmentStatus === 'misaligned' ? '!' : '✓'}</span>
                <span>{smartDetectionMsg || "Ranges aligned."}</span>
              </div>
            )}

            <div className="checkbox" style={{ marginBottom: 'var(--space-sm)' }}>
              <input
                type="checkbox"
                checked={analyzeSequentially}
                onChange={async e => {
                  const isChecked = e.target.checked;
                  setAnalyzeSequentially(isChecked);

                  if (isChecked) {
                    let status = alignmentStatus;
                    // If status is unknown but we have ranges, try to detect it now
                    if (!status && headerAddress && contentAddress) {
                      status = await checkAlignment(headerAddress, contentAddress);
                    }

                    if (status === 'aligned-row') {
                      setSequenceMode('row');
                      setOutputLocation('right');
                    } else if (status === 'aligned-col') {
                      setSequenceMode('col');
                      setOutputLocation('below');
                    }
                  }
                }}
              />
              <span>Analyze Sequentially (one row/column at a time with one output per row/column)</span>
            </div>



            {analyzeSequentially && (
              <div style={{ marginLeft: 'var(--space-xl)', marginBottom: 'var(--space-md)' }}>
                <div style={{ display: 'flex', gap: 'var(--space-md)' }}>
                  <label style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-xs)', cursor: 'pointer' }}>
                    <input
                      type="radio"
                      name="sequenceMode"
                      value="row"
                      checked={sequenceMode === 'row'}
                      onChange={e => {
                        setSequenceMode(e.target.value);
                        setOutputLocation('right');
                      }}
                    />
                    <span>Row by Row</span>
                  </label>
                  <label style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-xs)', cursor: 'pointer' }}>
                    <input
                      type="radio"
                      name="sequenceMode"
                      value="col"
                      checked={sequenceMode === 'col'}
                      onChange={e => {
                        setSequenceMode(e.target.value);
                        setOutputLocation('below');
                      }}
                    />
                    <span>Column by Column</span>
                  </label>
                </div>
                {alignmentStatus === 'aligned-row' && (
                  <p style={{ fontSize: 'var(--font-size-xs)', color: 'var(--color-primary)', marginTop: 'var(--space-xs)' }}>
                    Recommended: Row by Row (based on alignment)
                  </p>
                )}
                {alignmentStatus === 'aligned-col' && (
                  <p style={{ fontSize: 'var(--font-size-xs)', color: 'var(--color-primary)', marginTop: 'var(--space-xs)' }}>
                    Recommended: Column by Column (based on alignment)
                  </p>
                )}
                {smartDetectionMsg && !headerAddress && (
                  <p style={{ fontSize: 'var(--font-size-xs)', color: 'var(--color-primary)', marginTop: 'var(--space-xs)' }}>
                    {smartDetectionMsg}
                  </p>
                )}
              </div>
            )}

          </div>
        )}

        {/* Main Prompt Card */}
        <div className="card" style={{ marginBottom: 'var(--space-lg)' }}>
          <h2 className="heading-3" style={{ marginBottom: 'var(--space-md)' }}>Your Query</h2>

          {/* Quick Prompts Dropdown */}
          {settings?.prompts && settings.prompts.length > 0 && (
            <div className="input-group">
              <label>Quick Select (Optional)</label>
              <select
                className="select"
                value={selectedPromptId}
                onChange={handlePromptSelect}
              >
                <option value="">-- Choose a saved prompt --</option>
                {settings.prompts.map(p => (
                  <option key={p.id} value={p.id}>{p.name || p.text.substring(0, 30) + '...'}</option>
                ))}
              </select>
            </div>
          )}
          {/* Save/Delete/Update Actions */}
          <div style={{ display: 'flex', gap: 'var(--space-sm)', alignItems: 'center', marginTop: 'var(--space-sm)', marginBottom: 'var(--space-md)', flexWrap: 'wrap' }}>
            {!isSaving && prompt && (
              <>
                <button
                  onClick={initiateSave}
                  className="btn btn-ghost btn-sm"
                  style={{ color: 'var(--color-primary)', fontSize: 'var(--font-size-xs)' }}
                >
                  Save as new
                </button>
                {selectedPromptId && (
                  <>
                    <button
                      onClick={updatePrompt}
                      className="btn btn-ghost btn-sm"
                      style={{ color: updateSuccess ? 'var(--color-success)' : 'var(--color-primary)', fontSize: 'var(--font-size-xs)' }}
                    >
                      {updateSuccess ? '✓ Saved!' : 'Update current'}
                    </button>
                    <button
                      onClick={initiateDelete}
                      className="btn btn-ghost btn-sm"
                      style={{ color: 'var(--color-text-tertiary)', fontSize: 'var(--font-size-xs)' }}
                      onMouseEnter={(e) => e.currentTarget.style.color = 'var(--color-error)'}
                      onMouseLeave={(e) => e.currentTarget.style.color = 'var(--color-text-tertiary)'}
                    >
                      Delete
                    </button>
                  </>
                )}
              </>
            )}
          </div>

          {/* Save Input (Conditional) */}
          {isSaving && (
            <div style={{ marginBottom: 'var(--space-md)', padding: 'var(--space-lg)', background: 'var(--color-bg-surface)', border: '1px solid var(--color-border-medium)', borderRadius: 'var(--radius-md)' }}>
              <label style={{ display: 'block', fontSize: 'var(--font-size-sm)', fontWeight: 'var(--font-weight-medium)', marginBottom: 'var(--space-sm)', color: 'var(--color-text-primary)' }}>Name your prompt</label>
              <div style={{ display: 'flex', gap: 'var(--space-sm)' }}>
                <input
                  className="input"
                  style={{ flex: 1 }}
                  placeholder="Enter a name..."
                  value={saveName}
                  onChange={(e) => setSaveName(e.target.value)}
                  autoFocus
                />
                <button className="btn btn-primary" onClick={confirmSave}>Save</button>
                <button className="btn btn-secondary" onClick={cancelSave}>Cancel</button>
              </div>
            </div>
          )}

          {/* Delete Confirmation (Conditional) */}
          {isDeleting && (
            <div style={{ marginBottom: 'var(--space-md)', padding: 'var(--space-md)', background: 'var(--color-error-bg)', border: '1px solid var(--color-error)', borderRadius: 'var(--radius-md)' }}>
              <p style={{ fontSize: 'var(--font-size-base)', color: 'var(--color-text-primary)', fontWeight: 'var(--font-weight-medium)', marginBottom: 'var(--space-md)' }}>Delete "{settings.prompts.find(p => p.id.toString() === selectedPromptId)?.name}"?</p>
              <div style={{ display: 'flex', gap: 'var(--space-sm)', justifyContent: 'flex-end' }}>
                <button className="btn btn-danger" onClick={confirmDelete}>Delete</button>
                <button className="btn btn-secondary" onClick={cancelDelete}>Cancel</button>
              </div>
            </div>
          )}



          {/* Prompt Input */}
          <div className="input-group">
            <label>{queryMode === 'table' ? 'What do you want to know about this table?' : 'What do you want to know about each cell?'}</label>
            <textarea
              className="textarea"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder={queryMode === 'table'
                ? "Example: Analyze the trends in this data\nExample: Summarize the key findings\nExample: Identify anomalies in the values"
                : "Example: Summarize this text in one sentence\nExample: Perform a sentiment analysis\nExample: Extract the email address"}
              style={{ minHeight: '140px' }}
            />
            <p className="input-hint">
              {queryMode === 'table'
                ? "The AI will analyze the table data based on this prompt"
                : "Each selected cell will be processed individually with this prompt"}
            </p>
          </div>

          {/* Output Location (Both Modes) */}
          <div className="input-group">
            <label>Where to place results</label>
            <select
              className="select"
              value={outputLocation}
              onChange={(e) => setOutputLocation(e.target.value)}
            >
              <option value="right">Column right of table</option>
              <option value="below">Row below table</option>
              <option value="new-sheet">New worksheet</option>
            </select>
          </div>

          {/* Knowledge Base Toggle */}
          <div className="checkbox" style={{ marginTop: 'var(--space-md)' }}>
            <input
              type="checkbox"
              checked={useRag}
              onChange={(e) => {
                setUseRag(e.target.checked);
                if (!e.target.checked) setSourceOnly(false); // Reset source only when KB disabled
              }}
            />
            <span>Use Knowledge Base (search uploaded documents for context)</span>
          </div>

          {useRag && (
            <div className="checkbox" style={{ marginLeft: 'var(--space-xl)', paddingLeft: 'var(--space-md)', borderLeft: '2px solid var(--color-border-medium)' }}>
              <input
                type="checkbox"
                checked={sourceOnly}
                onChange={(e) => setSourceOnly(e.target.checked)}
              />
              <span>Return source context only (skip AI processing for faster results)</span>
            </div>
          )}

          {useRag && queryMode === 'table' && (
            <div className="input-group" style={{ marginLeft: 'var(--space-xl)', paddingLeft: 'var(--space-md)', borderLeft: '2px solid var(--color-border-medium)', marginTop: 'var(--space-sm)' }}>
              <label>Search Refinement Strategy</label>
              <select
                className="select"
                value={refinementStrategy}
                onChange={(e) => setRefinementStrategy(e.target.value)}
              >
                <option value="none">No Refinement (Use raw cell text)</option>
                <option value="auto">Auto-Refine (AI generates query)</option>
                <option value="review">Review Each (Approve queries before search)</option>
              </select>
              <p className="input-hint">
                Based on your query and the table, what are we looking for?
              </p>
            </div>
          )}
        </div>

        {/* Process Button */}
        <div style={{ display: 'flex', gap: 'var(--space-sm)' }}>
          <button
            className="btn btn-primary"
            style={{ flex: 1, padding: 'var(--space-lg)', fontSize: 'var(--font-size-lg)', fontWeight: 'var(--font-weight-semibold)' }}
            onClick={runBatch}
            disabled={loading || !prompt}
          >
            {loading ? (
              <>
                <svg style={{ width: '24px', height: '24px', animation: 'spin 1s linear infinite' }} fill="none" viewBox="0 0 24 24">
                  <circle style={{ opacity: 0.25 }} cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path style={{ opacity: 0.75 }} fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                Processing...
              </>
            ) : (
              <>
                <svg style={{ width: '20px', height: '20px' }} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 10V3L4 14h7v7l9-11h-7z"></path>
                </svg>
                Process Selected Cells
              </>
            )}
          </button>



          {loading && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
              <button
                className="btn btn-danger"
                style={{ padding: 'var(--space-lg)', fontSize: 'var(--font-size-lg)', fontWeight: 'var(--font-weight-semibold)' }}
                onClick={handleCancel}
              >
                Cancel
              </button>
              <div style={{
                textAlign: 'center',
                fontSize: '12px',
                color: 'var(--color-text-secondary)',
                fontVariantNumeric: 'tabular-nums'
              }}>
                Timeout in: {Math.floor(Math.max(0, (settings?.config?.timeout_seconds || 1200) - elapsedSeconds) / 60)}:{(Math.max(0, (settings?.config?.timeout_seconds || 1200) - elapsedSeconds) % 60).toString().padStart(2, '0')}
              </div>
            </div>
          )}
        </div>

        {/* Status Message */}
        {status && (
          <div style={{
            marginTop: 'var(--space-md)',
            textAlign: 'center',
            fontSize: 'var(--font-size-base)',
            padding: 'var(--space-md)',
            borderRadius: 'var(--radius-md)',
            background: status.startsWith('Error') ? 'var(--color-error-bg)' : 'var(--color-success-bg)',
            color: status.startsWith('Error') ? 'var(--color-error)' : 'var(--color-success)',
            fontWeight: 'var(--font-weight-medium)'
          }}>
            {status}
          </div>
        )}

      </div>

      {/* Refinement Modal */}
      {
        showRefinementModal && (
          <div style={{
            position: 'fixed',
            inset: 0,
            background: 'rgba(0,0,0,0.5)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 1000
          }}>
            <div className="card" style={{ maxWidth: '800px', width: '90%', maxHeight: '80vh', display: 'flex', flexDirection: 'column', padding: 'var(--space-xl)' }}>
              <h3 className="heading-3" style={{ marginBottom: 'var(--space-md)' }}>Refine Search Queries</h3>
              <p style={{ color: 'var(--color-text-secondary)', marginBottom: 'var(--space-md)' }}>
                The AI has proposed search queries for your data. You can edit them below before searching the Knowledge Base.
              </p>

              <div style={{ flex: 1, overflowY: 'auto', marginBottom: 'var(--space-lg)', border: '1px solid var(--color-border-light)', borderRadius: 'var(--radius-md)' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 'var(--font-size-sm)' }}>
                  <thead style={{ background: 'var(--color-bg-surface)', position: 'sticky', top: 0 }}>
                    <tr>
                      <th style={{ padding: 'var(--space-sm)', textAlign: 'left', borderBottom: '1px solid var(--color-border-light)', width: '15%' }}>Row</th>
                      <th style={{ padding: 'var(--space-sm)', textAlign: 'left', borderBottom: '1px solid var(--color-border-light)' }}>Search Query</th>
                    </tr>
                  </thead>
                  <tbody>
                    {proposedQueries.map((q, i) => (
                      <tr key={i} style={{ borderBottom: '1px solid var(--color-border-light)' }}>
                        <td style={{ padding: 'var(--space-sm)', verticalAlign: 'top', color: 'var(--color-text-secondary)', fontWeight: 'bold' }}>
                          Row {i + 1}
                        </td>
                        <td style={{ padding: 'var(--space-sm)', verticalAlign: 'top' }}>
                          <textarea
                            className="textarea"
                            value={q}
                            onChange={(e) => {
                              const newQueries = [...proposedQueries];
                              newQueries[i] = e.target.value;
                              setProposedQueries(newQueries);
                            }}
                            style={{ minHeight: '60px', width: '100%', fontSize: 'var(--font-size-sm)' }}
                          />
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              <div style={{ display: 'flex', gap: 'var(--space-sm)', justifyContent: 'flex-end' }}>
                <button className="btn btn-ghost" onClick={cancelRefinement}>Cancel</button>
                <button className="btn btn-primary" onClick={confirmRefinement}>Search & Process All</button>
              </div>
            </div>
          </div>
        )
      }

      {/* Overwrite Confirmation Modal */}
      {
        showOverwriteConfirm && (
          <div style={{
            position: 'fixed',
            inset: 0,
            background: 'rgba(0,0,0,0.5)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 1000
          }}>
            <div className="card" style={{ maxWidth: '400px', padding: 'var(--space-xl)' }}>
              <h3 className="heading-3" style={{ marginBottom: 'var(--space-md)' }}>Overwrite Existing Data?</h3>
              <p style={{ color: 'var(--color-text-secondary)', marginBottom: 'var(--space-lg)' }}>
                The output cells contain data. Do you want to overwrite them with the new results?
              </p>
              <div style={{ display: 'flex', gap: 'var(--space-sm)', justifyContent: 'flex-end' }}>
                <button className="btn btn-ghost" onClick={cancelOverwrite}>Cancel</button>
                <button className="btn btn-primary" onClick={confirmOverwrite}>Overwrite</button>
              </div>
            </div>
          </div>
        )
      }
    </div >
  );
};



const ChatView = ({ settings, handleViewSource, activeDocs = [], onProcessingChange }) => {
  const [messages, setMessages] = useState(() => {
    // Load messages from localStorage on mount
    const saved = localStorage.getItem('chatMessages');
    return saved ? JSON.parse(saved) : [];
  });
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [showClearConfirm, setShowClearConfirm] = useState(false);
  const [showOverwriteConfirm, setShowOverwriteConfirm] = useState(false);
  const [chatUseRag, setChatUseRag] = useState(false);
  const [chatAttachedData, setChatAttachedData] = useState(null); // {text: string, summary: string }
  const [pendingExportText, setPendingExportText] = useState(null);
  const messagesEndRef = useRef(null);
  const abortControllerRef = useRef(null);

  // Save messages to localStorage whenever they change
  useEffect(() => {
    localStorage.setItem('chatMessages', JSON.stringify(messages));
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Sync loading state with parent
  useEffect(() => {
    if (onProcessingChange) {
      onProcessingChange(loading);
    }
  }, [loading, onProcessingChange]);

  const handleClearChat = () => {
    setShowClearConfirm(true);
  };

  const confirmClear = () => {
    setMessages([]);
    localStorage.removeItem('chatMessages');
    setShowClearConfirm(false);
  };

  const handleExportToExcel = async () => {
    if (messages.length === 0) {
      return;
    }

    const chatText = messages.map(msg => {
      const role = msg.role === 'user' ? 'You' : msg.role === 'assistant' ? 'AI' : 'System';
      return `[${role}]: ${msg.content}`;
    }).join('\n\n');

    await initiateExport(chatText);
  };

  const handleExportMessage = async (msg) => {
    if (!msg || !msg.content) return;
    await initiateExport(msg.content);
  };

  const initiateExport = async (text) => {
    if (!text) return;

    try {
      await Excel.run(async (context) => {
        const range = context.workbook.getSelectedRange();
        range.load('values');
        await context.sync();

        // Check if cell has content
        const currentValue = range.values[0][0];
        if (currentValue && currentValue.toString().trim() !== '') {
          // Store text for confirmation and show modal
          setPendingExportText(text);
          setShowOverwriteConfirm(true);
          return;
        }

        // If cell is empty, export directly
        await performExport(context, range, text);
      });
    } catch (e) {
      console.error('Export failed:', e);
    }
  };

  const performExport = async (context, range, text) => {
    range.values = [[text]];
    range.format.wrapText = true;
    range.format.verticalAlignment = "Top";
    await context.sync();
  };

  const confirmOverwrite = async () => {
    if (!pendingExportText) return;
    try {
      // Create a fresh Excel context to export
      await Excel.run(async (context) => {
        const range = context.workbook.getSelectedRange();
        await performExport(context, range, pendingExportText);
      });
    } catch (e) {
      console.error('Export failed:', e);
    }
    setShowOverwriteConfirm(false);
    setPendingExportText(null);
  };

  const cancelOverwrite = () => {
    setShowOverwriteConfirm(false);
    setPendingExportText(null);
  };

  const handleAddSelection = async () => {
    try {
      await Excel.run(async (context) => {
        const range = context.workbook.getSelectedRange();
        range.load("address, rowCount, columnCount");
        await context.sync();

        // Limit size to prevent huge context
        if (range.rowCount * range.columnCount > 10000) {
          alert("Selection is too large. Please select a smaller range.");
          return;
        }

        // Reuse the extraction logic from QueryView (we need to move it up or duplicate it)
        // Since it's defined inside QueryView, we can't access it here.
        // We should probably move extractFormattedTable to a shared helper or define it here.
        // For now, let's duplicate the core logic for simplicity and speed.

        range.load("values");
        await context.sync();

        const values = range.values;
        if (!values || values.length === 0) return;

        // Simple markdown table generation
        let markdown = "| " + values[0].map(v => String(v)).join(" | ") + " |\n";
        markdown += "|" + values[0].map(() => "---").join("|") + "|\n";
        for (let i = 1; i < values.length; i++) {
          markdown += "| " + values[i].map(v => String(v)).join(" | ") + " |\n";
        }

        setChatAttachedData({
          text: markdown,
          summary: `Table: ${range.address}`
        });
      });
    } catch (e) {
      console.error("Failed to add selection", e);
      alert("Failed to add selection: " + e.message);
    }
  };

  const handleStop = async () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
      setLoading(false);

      // Explicitly request model unload to ensure backend stops processing
      try {
        const modelName = settings?.config?.model_name || 'llama3';
        await fetchAPI(`${API_BASE}/unload-model`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ model: modelName })
        });
      } catch (e) {
        console.error("Failed to send unload request:", e);
      }
    }
  };

  const handleSend = async () => {
    if (!input.trim()) return;

    const userMsg = {
      role: 'user',
      content: input,
      context_data: chatAttachedData ? chatAttachedData.text : null,
      context_summary: chatAttachedData ? chatAttachedData.summary : null
    };

    setMessages(prev => [...prev, userMsg, { role: 'assistant', content: '', sources: null }]);
    setInput('');
    setLoading(true);

    // Create new AbortController
    const controller = new AbortController();
    abortControllerRef.current = controller;

    try {
      const res = await fetchAPI(`${API_BASE}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: [...messages, userMsg],
          model: settings?.config?.model_name || 'llama3',
          use_rag: chatUseRag,
          filtered_documents: activeDocs.length > 0 ? activeDocs : undefined
          // context_data is now inside the message object
        }),
        signal: controller.signal
      });

      if (!res.ok) throw new Error(res.statusText);

      // Check for Sources header
      const sourcesHeader = res.headers.get('X-Sources');
      let sources = null;
      if (sourcesHeader) {
        try {
          sources = JSON.parse(sourcesHeader);
          // Update the last message (assistant placeholder) with sources
          setMessages(prev => {
            const newMessages = [...prev];
            const lastMsg = { ...newMessages[newMessages.length - 1] };
            lastMsg.sources = sources;
            newMessages[newMessages.length - 1] = lastMsg;
            return newMessages;
          });
        } catch (e) {
          console.error("Failed to parse sources header", e);
        }
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let done = false;
      let currentContent = '';

      while (!done) {
        const { value, done: doneReading } = await reader.read();
        done = doneReading;
        const chunkValue = decoder.decode(value, { stream: true });

        currentContent += chunkValue;

        setMessages(prev => {
          const newHistory = [...prev];
          const lastMsg = newHistory[newHistory.length - 1];
          newHistory[newHistory.length - 1] = { ...lastMsg, content: currentContent };
          return newHistory;
        });
      }
    } catch (e) {
      if (e.name === 'AbortError') {
        setMessages(prev => [...prev, { role: 'system', content: 'Generation stopped by user.' }]);
      } else {
        setMessages(prev => [...prev, { role: 'system', content: `Error: ${e.message}` }]);
      }
    } finally {
      setLoading(false);
      abortControllerRef.current = null;
      // Clear attached data after sending
      setChatAttachedData(null);
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', background: 'transparent', overflowY: 'auto', position: 'relative' }}>
      {/* Spacer for the absolute app header */}
      <div style={{ height: '44px', flexShrink: 0 }} />


      {/* Messages Area */}
      <div style={{ flex: 1, padding: 'var(--space-lg)', display: 'flex', flexDirection: 'column', gap: 'var(--space-lg)' }}>
        {messages.length === 0 && (
          <div style={{
            textAlign: 'center',
            marginTop: 'var(--space-3xl)',
            padding: 'var(--space-2xl)'
          }}>
            <div style={{
              width: '64px',
              height: '64px',
              background: 'var(--color-bg-hover)',
              borderRadius: '50%',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              margin: '0 auto var(--space-lg)'
            }}>
              <svg style={{ width: '32px', height: '32px', color: 'var(--color-text-tertiary)' }} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z"></path>
              </svg>
            </div>
            <p style={{ fontSize: 'var(--font-size-lg)', fontWeight: 'var(--font-weight-semibold)', color: 'var(--color-text-primary)', marginBottom: 'var(--space-xs)' }}>
              No messages yet
            </p>
            <p style={{ fontSize: 'var(--font-size-sm)', color: 'var(--color-text-secondary)' }}>
              Start a conversation with your data
            </p>
          </div>
        )}

        {messages.map((msg, i) => (
          <div key={i} style={{
            display: 'flex',
            width: '100%',
            justifyContent: msg.role === 'user' ? 'flex-end' : 'flex-start',
            alignItems: 'center',
            gap: 'var(--space-md)',
            position: 'relative'
          }}>
            {/* Export Button - Left side for User bubble */}
            {msg.role === 'user' && !loading && msg.content && msg.role !== 'system' && (
              <button
                className="btn btn-ghost btn-sm"
                style={{
                  width: '24px',
                  height: '24px',
                  padding: 0,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  borderRadius: '50%',
                  background: 'rgba(255,255,255,0.4)',
                  backdropFilter: 'blur(8px)',
                  border: '1px solid rgba(255,255,255,0.4)',
                  color: 'var(--color-primary)',
                  boxShadow: '0 2px 8px rgba(0,0,0,0.05)',
                  transition: 'all 0.2s',
                  flexShrink: 0
                }}
                onClick={() => handleExportMessage(msg)}
                title="Export this message to Excel cell"
              >
                <svg style={{ width: '12px', height: '12px' }} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path>
                </svg>
              </button>
            )}

            <div style={{
              maxWidth: '80%',
              background: msg.role === 'user' ? 'rgba(0, 107, 255, 0.85)' : 'rgba(255, 255, 255, 0.3)',
              backdropFilter: 'blur(10px)',
              WebkitBackdropFilter: 'blur(10px)',
              border: '1px solid rgba(255, 255, 255, 0.2)',
              color: msg.role === 'user' ? 'white' : 'var(--color-text-primary)',
              padding: 'var(--space-md)',
              borderRadius: 'var(--radius-lg)',
              borderBottomRightRadius: msg.role === 'user' ? 'var(--radius-xs)' : 'var(--radius-lg)',
              borderBottomLeftRadius: msg.role !== 'user' ? 'var(--radius-xs)' : 'var(--radius-lg)',
              boxShadow: '0 4px 12px rgba(0,0,0,0.03)',
              position: 'relative'
            }}>
              {/* Context Data Chip */}
              {msg.context_summary && (
                <div style={{
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: 'var(--space-xs)',
                  padding: '2px 6px',
                  background: 'rgba(255,255,255,0.2)',
                  borderRadius: 'var(--radius-sm)',
                  fontSize: 'var(--font-size-xs)',
                  marginBottom: 'var(--space-xs)',
                  color: 'inherit'
                }}>
                  <svg style={{ width: '10px', height: '10px' }} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M3 10h18M3 14h18m-9-4v8m-7 0h14a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"></path>
                  </svg>
                  {msg.context_summary}
                </div>
              )}

              <div style={{ whiteSpace: msg.role === 'user' ? 'pre-wrap' : 'normal', lineHeight: 1.5 }}>
                {(() => {
                  // Generic loading indicator for empty assistant message
                  if (msg.role === 'assistant' && !msg.content.trim() && loading && i === messages.length - 1) {
                    return (
                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--color-text-secondary)', fontStyle: 'italic' }}>
                        <div style={{
                          width: '12px',
                          height: '12px',
                          border: '2px solid var(--color-text-tertiary)',
                          borderTopColor: 'var(--color-primary)',
                          borderRadius: '50%',
                          animation: 'spin 1s linear infinite'
                        }}></div>
                        <span>Thinking...</span>
                      </div>
                    );
                  }

                  return (
                    <>
                      {msg.role === 'assistant' ? (
                        <Markdown
                          remarkPlugins={[remarkGfm]}
                          rehypePlugins={[rehypeRaw, rehypeSanitize]}
                          components={{
                            p: ({ node: _node, ...props }) => <p style={{ marginBottom: 'var(--space-sm)' }} {...props} />,
                            ul: ({ node: _node, ...props }) => <ul style={{ marginLeft: 'var(--space-lg)', marginBottom: 'var(--space-sm)' }} {...props} />,
                            ol: ({ node: _node, ...props }) => <ol style={{ marginLeft: 'var(--space-lg)', marginBottom: 'var(--space-sm)' }} {...props} />,
                            code: ({ node: _node, inline, ...props }) =>
                              inline
                                ? <code style={{ background: 'rgba(0,0,0,0.1)', padding: '2px 4px', borderRadius: '3px', fontSize: '0.9em' }} {...props} />
                                : <code style={{ display: 'block', background: 'rgba(0,0,0,0.05)', padding: 'var(--space-sm)', borderRadius: 'var(--radius-sm)', fontSize: '0.9em', overflowX: 'auto' }} {...props} />,
                            h1: ({ node: _node, ...props }) => <h1 style={{ fontSize: 'var(--font-size-xl)', fontWeight: 'var(--font-weight-bold)', marginBottom: 'var(--space-sm)' }} {...props} />,
                            h2: ({ node: _node, ...props }) => <h2 style={{ fontSize: 'var(--font-size-lg)', fontWeight: 'var(--font-weight-semibold)', marginBottom: 'var(--space-sm)' }} {...props} />,
                            h3: ({ node: _node, ...props }) => <h3 style={{ fontSize: 'var(--font-size-md)', fontWeight: 'var(--font-weight-semibold)', marginBottom: 'var(--space-xs)' }} {...props} />,
                            table: ({ node: _node, ...props }) => <table style={{ borderCollapse: 'collapse', width: '100%', marginBottom: 'var(--space-md)', fontSize: 'var(--font-size-sm)' }} {...props} />,
                            thead: ({ node: _node, ...props }) => <thead style={{ background: 'var(--color-bg-hover)' }} {...props} />,
                            tbody: ({ node: _node, ...props }) => <tbody {...props} />,
                            tr: ({ node: _node, ...props }) => <tr style={{ borderBottom: '1px solid var(--color-border-light)' }} {...props} />,
                            th: ({ node: _node, ...props }) => <th style={{ padding: 'var(--space-sm)', textAlign: 'left', fontWeight: '600', border: '1px solid var(--color-border-light)' }} {...props} />,
                            td: ({ node: _node, ...props }) => <td style={{ padding: 'var(--space-sm)', border: '1px solid var(--color-border-light)' }} {...props} />,
                          }}
                        >
                          {msg.content}
                        </Markdown>
                      ) : (
                        msg.content
                      )}
                    </>
                  );
                })()}
              </div>

              {/* Sources Display */}
              {msg.sources && msg.sources.length > 0 && (
                <div style={{ marginTop: 'var(--space-md)', paddingTop: 'var(--space-sm)', borderTop: '1px solid var(--color-border-light)' }}>
                  <div style={{ fontSize: 'var(--font-size-xs)', fontWeight: '600', color: 'var(--color-text-secondary)', marginBottom: 'var(--space-xs)' }}>
                    Sources:
                  </div>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: 'var(--space-xs)' }}>
                    {msg.sources.map((source, idx) => (
                      <button
                        key={idx}
                        onClick={() => handleViewSource(source.source, source.text, source.id)}
                        style={{
                          fontSize: 'var(--font-size-xs)',
                          padding: '2px 8px',
                          background: 'var(--color-bg-surface)',
                          border: '1px solid var(--color-border-medium)',
                          borderRadius: 'var(--radius-full)',
                          color: 'var(--color-text-primary)',
                          cursor: 'pointer',
                          display: 'flex',
                          alignItems: 'center',
                          gap: '4px',
                          transition: 'all 0.2s'
                        }}
                        onMouseEnter={(e) => e.currentTarget.style.borderColor = 'var(--color-primary)'}
                        onMouseLeave={(e) => e.currentTarget.style.borderColor = 'var(--color-border-medium)'}
                      >
                        <span style={{ maxWidth: '150px', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                          {source.source}
                        </span>
                        <span style={{ opacity: 0.6 }}>
                          ({Math.round(source.score * 100)}%)
                        </span>
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </div>

            {/* Export Button - Right side for Assistant bubble */}
            {msg.role === 'assistant' && !loading && msg.content && msg.role !== 'system' && (
              <button
                className="btn btn-ghost btn-sm"
                style={{
                  width: '24px',
                  height: '24px',
                  padding: 0,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  borderRadius: '50%',
                  background: 'rgba(255,255,255,0.4)',
                  backdropFilter: 'blur(8px)',
                  border: '1px solid rgba(255,255,255,0.4)',
                  color: 'var(--color-primary)',
                  boxShadow: '0 2px 8px rgba(0,0,0,0.05)',
                  transition: 'all 0.2s',
                  flexShrink: 0
                }}
                onClick={() => handleExportMessage(msg)}
                title="Export this message to Excel cell"
              >
                <svg style={{ width: '12px', height: '12px' }} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path>
                </svg>
              </button>
            )}
          </div>
        ))}




        <div ref={messagesEndRef} />
      </div >

      <div className="glass-panel-bottom" style={{
        position: 'sticky',
        bottom: 0,
        flexShrink: 0,
        padding: 'var(--space-lg)',
        zIndex: 60
      }}>
        {chatAttachedData && (
          <div style={{
            display: 'inline-flex',
            alignItems: 'center',
            gap: 'var(--space-xs)',
            padding: '4px 8px',
            background: 'var(--color-primary-light)',
            color: 'var(--color-primary)',
            borderRadius: 'var(--radius-sm)',
            fontSize: 'var(--font-size-xs)',
            marginBottom: 'var(--space-sm)',
            fontWeight: 'var(--font-weight-medium)'
          }}>
            <svg style={{ width: '12px', height: '12px' }} fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M3 10h18M3 14h18m-9-4v8m-7 0h14a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"></path>
            </svg>
            {chatAttachedData.summary}
            <button
              onClick={() => setChatAttachedData(null)}
              style={{
                background: 'none',
                border: 'none',
                cursor: 'pointer',
                padding: '2px',
                marginLeft: '4px',
                color: 'var(--color-primary)',
                opacity: 0.7
              }}
              onMouseEnter={(e) => e.target.style.opacity = 1}
              onMouseLeave={(e) => e.target.style.opacity = 0.7}
            >
              <svg style={{ width: '12px', height: '12px' }} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12"></path>
              </svg>
            </button>
          </div>
        )}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-md)' }}>
          {/* Top Row: Multi-line Input */}
          <textarea
            className="input"
            style={{
              width: '100%',
              minHeight: '60px',
              maxHeight: '150px',
              padding: 'var(--space-md)',
              lineHeight: '1.5',
              resize: 'none',
              overflowY: 'auto'
            }}
            placeholder={chatUseRag ? "Ask a question about your documents..." : "Type a message..."}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                handleSend();
              }
            }}
            disabled={loading}
          />

          {/* Bottom Row: Actions & Send */}
          <div style={{ display: 'flex', gap: 'var(--space-sm)', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap' }}>
            <div style={{ display: 'flex', gap: 'var(--space-sm)', alignItems: 'center' }}>
              <label className="checkbox" style={{
                margin: 0,
                display: 'flex',
                alignItems: 'center',
                gap: 'var(--space-xs)',
                cursor: 'pointer',
                padding: '0 var(--space-md)',
                height: '34px',
                background: 'rgba(255,255,255,0.3)',
                borderRadius: 'var(--radius-sm)',
                border: '1px solid var(--color-border-light)'
              }}>
                <input
                  type="checkbox"
                  checked={chatUseRag}
                  onChange={(e) => setChatUseRag(e.target.checked)}
                  id="chat-use-kb-bottom"
                  style={{ width: '14px', height: '14px', cursor: 'pointer' }}
                />
                <span style={{ fontSize: '11px', fontWeight: 'bold', textTransform: 'uppercase', color: 'var(--color-secondary)' }}>Docs</span>
              </label>

              <div style={{ width: '1px', height: '20px', background: 'var(--color-border-light)', margin: '0 4px' }} />

              <div style={{ display: 'flex', gap: 'var(--space-xs)', alignItems: 'center' }}>
                <button
                  className="btn btn-secondary btn-sm"
                  onClick={handleAddSelection}
                  title="Attach selected table data"
                  style={{ width: '34px', height: '34px', padding: 0, display: 'flex', alignItems: 'center', justifyContent: 'center' }}
                >
                  <svg style={{ width: '16px', height: '16px' }} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 4v16m8-8H4"></path>
                  </svg>
                </button>
                <button
                  className="btn btn-secondary btn-sm"
                  onClick={handleExportToExcel}
                  disabled={messages.length === 0}
                  title="Export full chat to Excel"
                  style={{ width: '34px', height: '34px', padding: 0, display: 'flex', alignItems: 'center', justifyContent: 'center' }}
                >
                  <svg style={{ width: '16px', height: '16px' }} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path>
                  </svg>
                </button>
                <button
                  className="btn btn-secondary btn-sm"
                  onClick={handleClearChat}
                  disabled={messages.length === 0}
                  title="Clear chat history"
                  style={{ width: '34px', height: '34px', padding: 0, display: 'flex', alignItems: 'center', justifyContent: 'center' }}
                  onMouseEnter={(e) => messages.length > 0 && (e.currentTarget.style.color = 'var(--color-error)')}
                  onMouseLeave={(e) => messages.length > 0 && (e.currentTarget.style.color = 'inherit')}
                >
                  <svg style={{ width: '16px', height: '16px' }} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"></path>
                  </svg>
                </button>
              </div>
            </div>

            <button
              className="btn btn-primary"
              onClick={loading ? handleStop : handleSend}
              disabled={!loading && !input.trim()}
              style={{
                height: '34px',
                padding: '0 var(--space-lg)',
                minWidth: '100px',
                ...(loading ? { backgroundColor: 'var(--color-error)', borderColor: 'var(--color-error)' } : {})
              }}
            >
              {loading ? (
                <>
                  <svg style={{ width: '16px', height: '16px', marginRight: '8px' }} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <rect x="6" y="6" width="12" height="12" rx="2" ry="2"></rect>
                  </svg>
                  Stop
                </>
              ) : (
                <>
                  <svg style={{ width: '16px', height: '16px', marginRight: '8px' }} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"></path>
                  </svg>
                  Send
                </>
              )}
            </button>
          </div>
        </div>
      </div>

      {/* Clear Chat Confirmation Modal */}
      {showClearConfirm && (
        <div style={{
          position: 'fixed',
          inset: 0,
          background: 'rgba(0,0,0,0.5)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 1000
        }}>
          <div className="card" style={{ maxWidth: '400px', padding: 'var(--space-xl)' }}>
            <h3 className="heading-3" style={{ marginBottom: 'var(--space-md)' }}>Clear Chat History?</h3>
            <p style={{ color: 'var(--color-text-secondary)', marginBottom: 'var(--space-lg)' }}>
              Are you sure you want to clear all messages? This action cannot be undone.
            </p>
            <div style={{ display: 'flex', gap: 'var(--space-sm)', justifyContent: 'flex-end' }}>
              <button className="btn btn-ghost" onClick={() => setShowClearConfirm(false)}>Cancel</button>
              <button className="btn btn-danger" onClick={confirmClear}>Clear Chat</button>
            </div>
          </div>
        </div>
      )}

      {/* Overwrite Confirmation Modal */}
      {showOverwriteConfirm && (
        <div style={{
          position: 'fixed',
          inset: 0,
          background: 'rgba(0,0,0,0.5)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 1000
        }}>
          <div className="card" style={{ maxWidth: '400px', padding: 'var(--space-xl)' }}>
            <h3 className="heading-3" style={{ marginBottom: 'var(--space-md)' }}>Overwrite Cell?</h3>
            <p style={{ color: 'var(--color-text-secondary)', marginBottom: 'var(--space-lg)' }}>
              The selected cell contains data. Do you want to overwrite it with this export?
            </p>
            <div style={{ display: 'flex', gap: 'var(--space-sm)', justifyContent: 'flex-end' }}>
              <button className="btn btn-ghost" onClick={cancelOverwrite}>Cancel</button>
              <button className="btn btn-primary" onClick={confirmOverwrite}>Overwrite</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

const KnowledgeView = ({ documents, refreshDocs, excludedDocs = [], setExcludedDocs = () => { }, showInspectButton = false, onProcessingChange }) => {
  const [activeUploads, setActiveUploads] = useState({});
  const [fileToDelete, setFileToDelete] = useState(null);

  // Sync processing state with parent based on active uploads
  useEffect(() => {
    if (onProcessingChange) {
      // Check if any upload is active (not completed, not error, not warning)
      const isUploading = Object.values(activeUploads).some(
        upload => !upload.completed && !upload.error && !upload.warning
      );
      onProcessingChange(isUploading);
    }
  }, [activeUploads, onProcessingChange]);

  const handleUpload = async (e) => {
    const files = Array.from(e.target.files);
    if (files.length === 0) return;

    const newUploads = {};
    const allowedExtensions = ['.txt', '.docx', '.pdf', '.xlsx', '.xls', '.csv', '.md'];

    files.forEach(file => {
      if (activeUploads[file.name]) return;

      const ext = "." + file.name.split('.').pop().toLowerCase();

      if (documents.includes(file.name)) {
        newUploads[file.name] = { progress: 100, status: 'Skipped: Already exists', error: false, warning: true };
      } else if (!allowedExtensions.includes(ext)) {
        newUploads[file.name] = { progress: 100, status: 'Skipped: Unsupported type', error: false, warning: true };
      } else {
        newUploads[file.name] = { progress: 0, status: 'Starting...', error: false };
      }
    });

    if (Object.keys(newUploads).length === 0) {
      e.target.value = '';
      return;
    }

    setActiveUploads(prev => ({ ...prev, ...newUploads }));

    // Handle removal of skipped items
    Object.entries(newUploads).forEach(([name, state]) => {
      if (state.warning) {
        setTimeout(() => {
          setActiveUploads(prev => {
            const next = { ...prev };
            delete next[name];
            return next;
          });
        }, 5000);
      }
    });

    // Start actual uploads
    files.forEach(file => {
      if (newUploads[file.name] && !newUploads[file.name].warning) {
        uploadFile(file);
      }
    });

    e.target.value = '';
  };

  const uploadFile = async (file) => {
    const filename = file.name;
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetchAPI(`${API_BASE}/add-document`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Upload failed');
      }

      const { task_id } = await response.json();

      const pollInterval = setInterval(async () => {
        try {
          const statusRes = await fetchAPI(`${API_BASE}/task/${task_id}`);
          if (statusRes.ok) {
            const task = await statusRes.json();

            setActiveUploads(prev => ({
              ...prev,
              [filename]: {
                ...prev[filename],
                progress: task.progress,
                status: task.message
              }
            }));

            if (task.status === 'completed') {
              clearInterval(pollInterval);
              setActiveUploads(prev => ({
                ...prev,
                [filename]: { ...prev[filename], progress: 100, status: 'Completed', completed: true }
              }));
              refreshDocs();

              // Remove after delay
              setTimeout(() => {
                setActiveUploads(prev => {
                  const next = { ...prev };
                  delete next[filename];
                  return next;
                });
              }, 3000);

            } else if (task.status === 'failed') {
              clearInterval(pollInterval);
              setActiveUploads(prev => ({
                ...prev,
                [filename]: { ...prev[filename], status: `Error: ${task.message}`, error: true }
              }));
            }
          }
        } catch (e) {
          console.error("Polling error", e);
        }
      }, 1000);

    } catch (error) {
      console.error('Upload error:', error);
      setActiveUploads(prev => ({
        ...prev,
        [filename]: { ...prev[filename], status: `Error: ${error.message}`, error: true }
      }));
    }
  };

  const initiateDelete = (filename) => setFileToDelete(filename);
  const cancelDelete = () => setFileToDelete(null);

  const confirmDelete = async () => {
    if (!fileToDelete) return;
    try {
      await fetchAPI(`${API_BASE}/remove-document`, {
        method: 'DELETE',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ filename: fileToDelete })
      });
      await refreshDocs();
      setFileToDelete(null);
    } catch (e) {
      alert(`Error: ${e.message}`);
    }
  };

  const handleInspect = async (filename) => {
    try {
      const res = await fetchAPI(`${API_BASE}/document-chunks?filename=${encodeURIComponent(filename)}`);
      if (!res.ok) throw new Error("Failed to fetch chunks");
      const data = await res.json();
      const chunks = data.chunks;

      if (chunks.length === 0) {
        alert("No chunks found for this document.");
        return;
      }

      await Excel.run(async (context) => {
        const sheetName = `Inspect - ${filename.substring(0, 20)}`;
        let sheet;
        try {
          sheet = context.workbook.worksheets.add(sheetName);
        } catch {
          sheet = context.workbook.worksheets.add(`${sheetName} ${Date.now().toString().slice(-4)}`);
        }

        const range = sheet.getRange("A1:C1");
        range.values = [["Chunk ID", "Length", "Content"]];
        range.format.font.bold = true;

        const rows = chunks.map(c => [c.id, c.text.length, c.text]);
        const dataRange = sheet.getRange(`A2:C${rows.length + 1}`);
        dataRange.values = rows;

        sheet.getRange("C:C").format.wrapText = true;
        sheet.getRange("A:A").format.columnWidth = 60;
        sheet.getRange("B:B").format.columnWidth = 60;
        sheet.getRange("C:C").format.columnWidth = 400;
        dataRange.format.autofitColumns();
        sheet.getRange("C:C").format.columnWidth = 300;
        sheet.activate();
        await context.sync();
      });

    } catch (e) {
      console.error("Inspection failed", e);
      alert(`Inspection failed: ${e.message}`);
    }
  };

  const toggleDoc = (doc) => {
    if (excludedDocs.includes(doc)) {
      setExcludedDocs(excludedDocs.filter(d => d !== doc));
    } else {
      setExcludedDocs([...excludedDocs, doc]);
    }
  };

  return (
    <div style={{ height: '100%', overflowY: 'auto', padding: '44px var(--space-xs) var(--space-xs)', background: 'transparent' }}>
      <div style={{ height: '1px', background: 'rgba(255, 255, 255, 0.6)', position: 'fixed', top: '44px', left: 0, right: 0, zIndex: 1001, boxShadow: '0 2px 4px rgba(0,0,0,0.05)' }} />
      <div className="container" style={{ maxWidth: '100%' }}>

        {/* Header */}
        <div style={{ marginBottom: 'var(--space-2xl)' }}>
          <h1 className="heading-1" style={{ marginBottom: 'var(--space-xs)' }}>Knowledge Base</h1>
          <p className="text-secondary" style={{ fontSize: 'var(--font-size-base)', margin: 0 }}>
            Upload and manage your documents for AI-powered analysis
          </p>
        </div>

        {/* Upload Area */}
        <div className="card" style={{
          marginBottom: 'var(--space-xl)',
          border: '2px dashed var(--color-border-medium)',
          background: 'var(--color-bg-base)',
          cursor: 'pointer',
          position: 'relative',
          transition: 'all var(--transition-base)'
        }}>
          <input
            type="file"
            multiple
            onChange={handleUpload}
            style={{
              position: 'absolute',
              top: 0,
              left: 0,
              width: '100%',
              height: '100%',
              opacity: 0,
              cursor: 'pointer'
            }}
            id="file-upload"
            accept=".txt,.docx,.pdf,.xlsx,.xls,.csv,.md"
          />
          <div style={{ padding: 'var(--space-xl)', textAlign: 'center' }}>
            <div style={{
              fontSize: 'var(--font-size-lg)',
              fontWeight: 'var(--font-weight-semibold)',
              color: 'var(--color-primary)',
              marginBottom: 'var(--space-xs)'
            }}>
              Upload Documents
            </div>
            <p className="text-secondary" style={{ margin: 0, fontSize: 'var(--font-size-sm)' }}>
              Click to browse or drag files here (PDF, Word, Excel, etc.)
            </p>

            {/* Active Uploads List */}
            {Object.keys(activeUploads).length > 0 && (
              <div style={{ marginTop: 'var(--space-md)', textAlign: 'left' }}>
                {Object.entries(activeUploads).map(([filename, state]) => (
                  <div key={filename} style={{ marginBottom: 'var(--space-sm)', fontSize: 'var(--font-size-sm)' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                      <span style={{ fontWeight: 500, maxWidth: '200px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{filename}</span>
                      <span style={{
                        color: state.error ? 'var(--color-error)' : (state.warning ? 'var(--color-warning, #f59e0b)' : 'var(--color-text-secondary)'),
                        fontSize: 'var(--font-size-xs)'
                      }}>
                        {state.status}
                      </span>
                    </div>
                    <div style={{
                      height: '4px',
                      background: 'var(--color-bg-surface)',
                      borderRadius: '2px',
                      overflow: 'hidden'
                    }}>
                      <div style={{
                        height: '100%',
                        width: `${state.progress}%`,
                        background: state.error ? 'var(--color-error)' : (state.warning ? 'var(--color-warning, #f59e0b)' : (state.completed ? 'var(--color-success)' : 'var(--color-primary)')),
                        transition: 'width 0.3s ease'
                      }} />
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Document List */}
        <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
          <div style={{
            padding: 'var(--space-md) var(--space-lg)',
            borderBottom: '1px solid var(--color-border-light)',
            background: 'var(--color-bg-surface)',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center'
          }}>
            <h3 className="heading-3" style={{ margin: 0 }}>Documents ({documents.length})</h3>
            {documents.length > 0 && (
              <div style={{ fontSize: 'var(--font-size-sm)', color: 'var(--color-text-secondary)' }}>
                {documents.length - excludedDocs.length} Active
              </div>
            )}
          </div>

          {documents.length === 0 ? (
            <div style={{ padding: 'var(--space-xl)', textAlign: 'center', color: 'var(--color-text-secondary)' }}>
              No documents uploaded yet.
            </div>
          ) : (
            <ul style={{ listStyle: 'none', margin: 0, padding: 0 }}>
              {documents.map((doc, index) => (
                <li key={index} style={{
                  padding: 'var(--space-md) var(--space-lg)',
                  borderBottom: index < documents.length - 1 ? '1px solid var(--color-border-light)' : 'none',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  transition: 'background var(--transition-base)',
                  background: excludedDocs.includes(doc) ? 'var(--color-bg-base)' : 'white'
                }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-md)', flex: 1, overflow: 'hidden' }}>
                    <input
                      type="checkbox"
                      checked={!excludedDocs.includes(doc)}
                      onChange={() => toggleDoc(doc)}
                      style={{ cursor: 'pointer', width: '16px', height: '16px' }}
                    />
                    <div style={{
                      width: '6px',
                      height: '32px',
                      borderRadius: 'var(--radius-sm)',
                      background: excludedDocs.includes(doc) ? 'var(--color-border-light)' : 'var(--color-primary)',
                      flexShrink: 0,
                      transition: 'background var(--transition-base)'
                    }} />
                    <span style={{
                      fontWeight: 500,
                      whiteWhiteSpace: 'nowrap',
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                      color: excludedDocs.includes(doc) ? 'var(--color-text-secondary)' : 'var(--color-text-primary)'
                    }}>
                      {doc}
                    </span>
                  </div>

                  <div style={{ display: 'flex', gap: 'var(--space-sm)' }}>
                    {showInspectButton && (
                      <Button variant="ghost" onClick={() => handleInspect(doc)} style={{ fontSize: 'var(--font-size-sm)' }}>
                        Inspect
                      </Button>
                    )}
                    <Button variant="ghost" onClick={() => initiateDelete(doc)} style={{ fontSize: 'var(--font-size-sm)', color: 'var(--color-error)' }}>
                      Delete
                    </Button>
                  </div>
                </li>
              ))}
            </ul>
          )}
        </div>

        {/* Delete Confirmation Modal */}
        {fileToDelete && (
          <div style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            background: 'rgba(0,0,0,0.5)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 1000
          }}>
            <div className="card" style={{ width: '90%', maxWidth: '400px', padding: 'var(--space-xl)' }}>
              <h3 className="heading-3" style={{ marginTop: 0 }}>Delete Document?</h3>
              <p style={{ color: 'var(--color-text-secondary)', marginBottom: 'var(--space-xl)' }}>
                Are you sure you want to delete <strong>{fileToDelete}</strong>? This action cannot be undone.
              </p>
              <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 'var(--space-md)' }}>
                <button className="btn btn-secondary" onClick={cancelDelete}>Cancel</button>
                <button className="btn btn-danger" onClick={confirmDelete}>Delete</button>
              </div>
            </div>
          </div>
        )}

      </div>
    </div>
  );
};

const AuditView = () => {
  // const [selectedText, setSelectedText] = useState('');
  const [, setSelectedText] = useState('');
  const [sources, setSources] = useState([]);
  const [loading, setLoading] = useState(false);

  // New state for embedded view
  const [activeChunk, setActiveChunk] = useState(null);
  const [documentCache, setDocumentCache] = useState({});
  const [loadingContent, setLoadingContent] = useState(false);

  const checkSelection = async () => {
    try {
      await Excel.run(async (context) => {
        const range = context.workbook.getSelectedRange();
        range.load("values");
        await context.sync();

        const text = range.values[0][0];
        setSelectedText(text);

        const match = text.toString().match(/\(Source ID: (.*)\)/);

        if (match) {
          const idsPart = match[1];
          const idMatches = idsPart.split(',').map(s => {
            const m = s.trim().match(/^(\d+)\s*\((\d+)%\)$/);
            return m ? { id: m[1], score: m[2] } : null;
          }).filter(Boolean);

          if (idMatches.length > 0) {
            setLoading(true);
            const fetchedSources = [];
            for (const item of idMatches) {
              try {
                const res = await fetchAPI(`${API_BASE}/chunk/${item.id}`);
                if (res.ok) {
                  const chunk = await res.json();
                  fetchedSources.push({ ...chunk, score: item.score });
                }
              } catch {
                console.error("Failed to fetch chunk", item.id);
              }
            }
            setSources(fetchedSources);
            setLoading(false);

            // Auto-select first chunk if none selected or not in new list
            if (fetchedSources.length > 0) {
              setActiveChunk(fetchedSources[0]);
            } else {
              setActiveChunk(null);
            }
          } else {
            setSources([]);
            setActiveChunk(null);
          }
        } else {
          setSources([]);
          setActiveChunk(null);
        }
      });
    } catch (e) {
      console.error("Audit check failed", e);
    }
  };

  useEffect(() => {
    checkSelection();
    const onSelectionChanged = () => checkSelection();
    Office.context.document.addHandlerAsync(Office.EventType.DocumentSelectionChanged, onSelectionChanged);
    return () => {
      Office.context.document.removeHandlerAsync(Office.EventType.DocumentSelectionChanged, onSelectionChanged);
    };
  }, []);

  // Fetch document content when active chunk changes
  useEffect(() => {
    const fetchContent = async () => {
      if (!activeChunk) return;
      const filename = activeChunk.source;

      if (documentCache[filename]) return; // Already cached

      setLoadingContent(true);
      try {
        const res = await fetchAPI(`${API_BASE}/document-content?filename=${encodeURIComponent(filename)}`);
        if (res.ok) {
          const data = await res.json();
          setDocumentCache(prev => ({ ...prev, [filename]: data.content }));
        }
      } catch (e) {
        console.error("Failed to fetch document content", e);
      } finally {
        setLoadingContent(false);
      }
    };

    fetchContent();
  }, [activeChunk, documentCache]);

  const activeContent = activeChunk ? documentCache[activeChunk.source] : null;

  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column', background: 'transparent', overflowY: 'auto', position: 'relative' }}>
      {/* Spacer for the absolute app header */}
      <div style={{ height: '44px', flexShrink: 0 }} />

      {/* Audit Header - Sticky below app header */}
      <div className="glass-panel glass-panel-top" style={{
        position: 'sticky',
        top: '44px',
        padding: '0',
        border: 'none',
        borderBottom: '1px solid rgba(255, 255, 255, 0.4)',
        boxShadow: 'var(--shadow-sm)',
        flexShrink: 0,
        zIndex: 60
      }}>
        {sources.length > 0 && (
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: 'var(--space-md)',
            overflowX: 'auto',
            overflowY: 'hidden',
            WebkitOverflowScrolling: 'touch',
            padding: 'var(--space-sm)'
          }}>
            {/* Minimal Source Chips */}
            {sources.map((source, i) => {
              const isActive = activeChunk && activeChunk.id === source.id;
              return (
                <button
                  key={i}
                  onClick={() => setActiveChunk(source)}
                  className="card"
                  style={{
                    minWidth: '160px',
                    maxWidth: '220px',
                    flexShrink: 0,
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'flex-start',
                    padding: 'var(--space-xs) var(--space-sm)',
                    gap: '2px',
                    cursor: 'pointer',
                    border: isActive ? '1px solid var(--color-primary)' : '1px solid rgba(255,255,255,0.3)',
                    background: isActive ? 'var(--color-primary)' : 'rgba(255,255,255,0.4)',
                    backdropFilter: isActive ? 'none' : 'blur(8px)',
                    borderRadius: 'var(--radius-md)',
                    transition: 'all var(--transition-fast)'
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', width: '100%', alignItems: 'center' }}>
                    <span style={{
                      fontSize: '10px',
                      fontWeight: 'var(--font-weight-bold)',
                      textTransform: 'uppercase',
                      letterSpacing: '0.5px',
                      color: isActive ? 'rgba(255,255,255,0.9)' : 'var(--color-text-secondary)'
                    }}>
                      Source {i + 1}
                    </span>
                    <span style={{
                      fontSize: '10px',
                      fontWeight: 'var(--font-weight-bold)',
                      background: isActive ? 'rgba(255,255,255,0.2)' : 'var(--color-bg-base)',
                      color: isActive ? 'white' : 'var(--color-primary)',
                      padding: '1px 4px',
                      borderRadius: '4px',
                      flexShrink: 0
                    }}>
                      {source.score}%
                    </span>
                  </div>
                  <span style={{
                    fontSize: '11px',
                    fontWeight: 'var(--font-weight-medium)',
                    color: isActive ? 'white' : 'var(--color-text-primary)',
                    whiteSpace: 'nowrap',
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                    width: '100%',
                    textAlign: 'left'
                  }} title={source.source}>
                    {source.source}
                  </span>
                </button>
              );
            })}
          </div>
        )}
      </div>

      {/* Main Content Area */}
      <div style={{ flex: 1, overflow: 'hidden', position: 'relative' }}>
        {loading ? (
          <div style={{
            position: 'absolute',
            inset: 0,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            background: 'rgba(255,255,255,0.8)',
            backdropFilter: 'blur(4px)',
            zIndex: 20
          }}>
            <div style={{
              width: '32px',
              height: '32px',
              border: '3px solid var(--color-border-light)',
              borderTopColor: 'var(--color-primary)',
              borderRadius: '50%',
              animation: 'spin 1s linear infinite'
            }}></div>
          </div>
        ) : sources.length === 0 ? (
          <div style={{
            height: '100%',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            padding: 'var(--space-3xl)',
            textAlign: 'center'
          }}>
            <div style={{
              width: '64px',
              height: '64px',
              background: 'var(--color-bg-hover)',
              borderRadius: '50%',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              marginBottom: 'var(--space-lg)'
            }}>
              <svg style={{ width: '32px', height: '32px', color: 'var(--color-text-tertiary)' }} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"></path>
              </svg>
            </div>
            <p className="body-text" style={{ color: 'var(--color-text-secondary)', marginBottom: 'var(--space-sm)' }}>
              Select a cell containing citations to audit.
            </p>
            <code style={{
              fontSize: 'var(--font-size-xs)',
              background: 'var(--color-bg-hover)',
              padding: 'var(--space-xs) var(--space-sm)',
              borderRadius: 'var(--radius-sm)',
              color: 'var(--color-text-tertiary)',
              fontFamily: 'monospace'
            }}>
              (Source ID: ...)
            </code>
          </div>
        ) : (
          <div style={{ height: '100%', padding: 'var(--space-sm)' }}>
            {loadingContent && !activeContent ? (
              <div style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <div style={{
                  width: '32px',
                  height: '32px',
                  border: '3px solid var(--color-border-light)',
                  borderTopColor: 'var(--color-primary)',
                  borderRadius: '50%',
                  animation: 'spin 1s linear infinite'
                }}></div>
              </div>
            ) : (
              <MarkdownViewer
                isOpen={true}
                variant="embedded"
                content={activeContent}
                highlightText={activeChunk ? activeChunk.text : ''}
                filename={activeChunk ? activeChunk.source : ''}
              />
            )}
          </div>
        )}
      </div>
    </div >
  );
};

const SettingsView = ({
  settings,
  refreshSettings,
  includeColors,
  setIncludeColors,
  includeStyles,
  setIncludeStyles,
  debugPrompts,
  setDebugPrompts,
  showInspectButton,
  setShowInspectButton,
  developerMode,
  setDeveloperMode,
  onProcessingChange
}) => {
  const [config, setConfig] = useState(settings?.config || {});
  const [saveStatus, setSaveStatus] = useState('idle');
  const [clickCount, setClickCount] = useState(0);

  const handleHeaderClick = () => {
    const newCount = clickCount + 1;
    setClickCount(newCount);
    if (newCount >= 5) {
      setDeveloperMode(!developerMode);
      setClickCount(0);
      // Optional: You could add a toast here if you had a toast component
      console.log(`Developer mode ${!developerMode ? 'enabled' : 'disabled'}`);
    }
  };
  const [availableModels, setAvailableModels] = useState([]);
  const [benchmarkResults, setBenchmarkResults] = useState({});
  const [isBenchmarking, setIsBenchmarking] = useState(false);
  const [benchmarkProgress, setBenchmarkProgress] = useState('');
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [showGettingStarted, setShowGettingStarted] = useState(false);
  const [showRecommendedModels, setShowRecommendedModels] = useState(false);

  const copyToClipboard = (text) => {
    navigator.clipboard.writeText(text).then(() => {
      // Could add a toast here, but for now simple feedback
    });
  };

  useEffect(() => {
    if (settings?.config) setConfig(settings.config);
  }, [settings]);

  useEffect(() => {
    const fetchModels = async () => {
      try {
        const res = await fetchAPI(`${API_BASE}/list-models`);
        if (res.ok) {
          const data = await res.json();
          setAvailableModels(data.models);
        }
      } catch (e) {
        console.error("Failed to fetch models", e);
      }
    };
    fetchModels();
  }, []);

  const save = async () => {
    setSaveStatus('saving');
    try {
      const res = await fetchAPI(`${API_BASE}/settings`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ...settings, config })
      });
      if (!res.ok) throw new Error(`Server error: ${res.status}`);
      await refreshSettings();
      setSaveStatus('success');
      setTimeout(() => setSaveStatus('idle'), 3000);
    } catch (e) {
      setSaveStatus('error');
      alert(`Error: ${e.message}`);
    }
  };

  const runBenchmark = async () => {
    if (availableModels.length === 0) return;

    setIsBenchmarking(true);
    if (onProcessingChange) onProcessingChange(true);
    setBenchmarkResults({});

    for (let i = 0; i < availableModels.length; i++) {
      const model = availableModels[i];
      setBenchmarkProgress(`Testing ${model} (${i + 1}/${availableModels.length})...`);

      try {
        const res = await fetchAPI(`${API_BASE}/benchmark-model`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ model })
        });

        if (res.ok) {
          const data = await res.json();
          setBenchmarkResults(prev => ({
            ...prev,
            [model]: data.tps
          }));
        }
      } catch (e) {
        console.error(`Benchmark failed for ${model}`, e);
      }
    }

    setIsBenchmarking(false);
    if (onProcessingChange) onProcessingChange(false);
    setBenchmarkProgress('');
  };

  return (
    <div style={{ height: '100%', overflowY: 'auto', padding: '44px var(--space-xs) var(--space-xs)', background: 'transparent' }}>
      <div style={{ height: '1px', background: 'rgba(255, 255, 255, 0.6)', position: 'fixed', top: '44px', left: 0, right: 0, zIndex: 1001, boxShadow: '0 2px 4px rgba(0,0,0,0.05)' }} />
      <div className="container" style={{ maxWidth: '100%' }}>

        {/* Header */}
        <div style={{ marginBottom: 'var(--space-2xl)' }}>
          <h1
            className="heading-1"
            style={{ marginBottom: 'var(--space-xs)', userSelect: 'none', cursor: 'default' }}
            onClick={handleHeaderClick}
          >
            Settings {developerMode && <span style={{ fontSize: '0.5em', verticalAlign: 'middle', background: 'var(--color-primary)', color: 'white', padding: '2px 6px', borderRadius: '4px' }}>DEV</span>}
          </h1>
          <p className="text-secondary" style={{ fontSize: 'var(--font-size-base)', margin: 0 }}>
            Configure your AI model and application preferences
          </p>
        </div>

        {/* Getting Started Guide */}
        <div className="card" style={{ marginBottom: 'var(--space-xl)', background: 'var(--color-bg-surface)', border: '2px solid var(--color-primary-light)' }}>
          <h3
            className="heading-3"
            style={{ cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 'var(--space-sm)', userSelect: 'none', margin: 0 }}
            onClick={() => setShowGettingStarted(!showGettingStarted)}
          >
            <span>{showGettingStarted ? '▼' : '▶'}</span>
            <span>Getting Started with Cellami</span>
          </h3>

          {showGettingStarted && (
            <div style={{ marginTop: 'var(--space-lg)' }}>
              <p className="text-secondary" style={{ fontSize: 'var(--font-size-base)', marginBottom: 'var(--space-lg)', lineHeight: 1.6 }}>
                Start by downloading Ollama, a free and open-source tool that lets you run AI models locally on your computer. Follow these simple steps to get started:
              </p>

              {/* Step 1 */}
              <div style={{ marginBottom: 'var(--space-xl)' }}>
                <h4 className="heading-4" style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-sm)', marginBottom: 'var(--space-sm)' }}>
                  <span style={{
                    background: 'var(--color-primary)',
                    color: 'white',
                    borderRadius: '50%',
                    width: '24px',
                    height: '24px',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    fontSize: 'var(--font-size-sm)',
                    fontWeight: 'var(--font-weight-bold)',
                    flexShrink: 0
                  }}>1</span>
                  Download & Install Ollama
                </h4>
                <p className="text-secondary" style={{ fontSize: 'var(--font-size-base)', marginBottom: 'var(--space-sm)', lineHeight: 1.6 }}>
                  Visit <a href="https://ollama.com" target="_blank" rel="noopener noreferrer" style={{ color: 'var(--color-primary)', textDecoration: 'underline' }}>ollama.com</a> and download the installer for your system.
                </p>
                <p className="text-secondary" style={{ fontSize: 'var(--font-size-sm)', fontStyle: 'italic', color: 'var(--color-text-tertiary)' }}>
                  Once installed, Ollama should run in the background automatically (as indicated by Ollama icon in your system tray/menu bar).
                </p>
              </div>

              {/* Step 2 */}
              <div style={{ marginBottom: 'var(--space-xl)' }}>
                <h4 className="heading-4" style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-sm)', marginBottom: 'var(--space-sm)' }}>
                  <span style={{
                    background: 'var(--color-primary)',
                    color: 'white',
                    borderRadius: '50%',
                    width: '24px',
                    height: '24px',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    fontSize: 'var(--font-size-sm)',
                    fontWeight: 'var(--font-weight-bold)',
                    flexShrink: 0
                  }}>2</span>
                  Download a Model
                </h4>
                <p className="text-secondary" style={{ fontSize: 'var(--font-size-base)', marginBottom: 'var(--space-sm)', lineHeight: 1.6 }}>
                  Open the Ollama app (ensure it resides in your system tray/menu bar), <strong>open your computer's Terminal</strong> and paste one of the recommended model commands below and press enter.
                </p>

                <div style={{
                  background: 'var(--color-bg-base)',
                  border: '1px solid var(--color-border-light)',
                  borderRadius: 'var(--radius-md)',
                  padding: 'var(--space-lg)',
                  marginBottom: 'var(--space-md)'
                }}>
                  <h5
                    style={{
                      fontSize: 'var(--font-size-base)',
                      fontWeight: 'var(--font-weight-semibold)',
                      marginBottom: 'var(--space-sm)',
                      color: 'var(--color-text-primary)',
                      cursor: 'pointer',
                      display: 'flex',
                      alignItems: 'center',
                      gap: 'var(--space-xs)'
                    }}
                    onClick={() => setShowRecommendedModels(!showRecommendedModels)}
                  >
                    <span>{showRecommendedModels ? '▼' : '▶'}</span> Recommended Models
                  </h5>

                  {showRecommendedModels && (
                    <>
                      <div style={{ marginBottom: 'var(--space-md)' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-sm)', marginBottom: 'var(--space-xs)' }}>
                          <p style={{ fontSize: 'var(--font-size-base)', fontWeight: 'var(--font-weight-medium)', margin: 0, color: 'var(--color-text-primary)' }}>
                            ⚡ <strong>gemma3:4b</strong>
                          </p>
                          <button
                            className='btn btn-secondary btn-sm'
                            style={{ padding: '2px 8px', fontSize: '10px' }}
                            onClick={() => copyToClipboard('ollama pull gemma3:4b')}
                            title="Copy install command"
                          >
                            Copy Install Command
                          </button>
                        </div>

                        <p className="text-secondary" style={{ fontSize: 'var(--font-size-sm)', marginLeft: 'var(--space-xl)', marginBottom: 'var(--space-xs)' }}>
                          Perfect for systems with limited memory (8GB or less).
                        </p>
                        <p className="text-secondary" style={{ fontSize: 'var(--font-size-xs)', marginLeft: 'var(--space-xl)', color: 'var(--color-text-tertiary)' }}>
                          <em>Examples: "Categorized these expenses as Travel, Food, or Utilities", "Extract email and sentiment from each customer review"</em><br />
                          <span style={{ color: 'var(--color-success)' }}>✅ Fast text processing</span><br />
                          <span style={{ color: 'var(--color-error)' }}>⚠️ Unreliable for math or numerical accuracy</span>
                        </p>
                        <p className="text-secondary" style={{ fontSize: 'var(--font-size-xs)', marginLeft: 'var(--space-xl)', fontStyle: 'italic', color: 'var(--color-text-tertiary)', marginTop: 'var(--space-xs)' }}>
                          Alternatives to try: <strong>ministral-3:3b</strong>
                        </p>
                      </div>

                      <div style={{ marginBottom: 'var(--space-md)' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-sm)', marginBottom: 'var(--space-xs)' }}>
                          <p style={{ fontSize: 'var(--font-size-base)', fontWeight: 'var(--font-weight-medium)', margin: 0, color: 'var(--color-text-primary)' }}>
                            🌟 <strong>ministral-3:8b</strong>
                          </p>
                          <button
                            className='btn btn-secondary btn-sm'
                            style={{ padding: '2px 8px', fontSize: '10px' }}
                            onClick={() => copyToClipboard('ollama pull ministral-3:8b')}
                            title="Copy install command"
                          >
                            Copy Install Command
                          </button>
                        </div>
                        <p className="text-secondary" style={{ fontSize: 'var(--font-size-sm)', marginLeft: 'var(--space-xl)', marginBottom: 'var(--space-xs)' }}>
                          Great balance of speed and quality. Works well on most systems.
                        </p>
                        <p className="text-secondary" style={{ fontSize: 'var(--font-size-xs)', marginLeft: 'var(--space-xl)', color: 'var(--color-text-tertiary)' }}>
                          <em>Examples: "Complete this RFP based on the Knowledge Base", "Generally summarize trends from each line of this table"</em><br />
                          <span style={{ color: 'var(--color-success)' }}>✅ General reasoning capabilities</span><br />
                          <span style={{ color: 'var(--color-error)' }}>⚠️ Limited math accuracy, moderate speed</span>
                        </p>
                        <p className="text-secondary" style={{ fontSize: 'var(--font-size-xs)', marginLeft: 'var(--space-xl)', fontStyle: 'italic', color: 'var(--color-text-tertiary)', marginTop: 'var(--space-xs)' }}>
                          Alternatives to try: <strong>gemma3:12b</strong>
                        </p>
                      </div>

                      <div>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-sm)', marginBottom: 'var(--space-xs)' }}>
                          <p style={{ fontSize: 'var(--font-size-base)', fontWeight: 'var(--font-weight-medium)', margin: 0, color: 'var(--color-text-primary)' }}>
                            🚀 <strong>gpt-oss:20b</strong>
                          </p>
                          <button
                            className='btn btn-secondary btn-sm'
                            style={{ padding: '2px 8px', fontSize: '10px' }}
                            onClick={() => copyToClipboard('ollama pull gpt-oss:20b')}
                            title="Copy install command"
                          >
                            Copy Install Command
                          </button>
                        </div>
                        <p className="text-secondary" style={{ fontSize: 'var(--font-size-sm)', marginLeft: 'var(--space-xl)', marginBottom: 'var(--space-xs)' }}>
                          Highest quality responses. Requires 32GB+ memory for optimal performance. Optimized for Apple Silicon.
                        </p>
                        <p className="text-secondary" style={{ fontSize: 'var(--font-size-xs)', marginLeft: 'var(--space-xl)', color: 'var(--color-text-tertiary)' }}>
                          <em>Examples: "What is driving financial trends based on the 10-k?", "Using a threshold of $5,500, flag journal entries that require attention."</em><br />
                          <span style={{ color: 'var(--color-success)' }}>✅ Capable of math and detailed analysis</span><br />
                          <span style={{ color: 'var(--color-error)' }}>⚠️ Slow if limited memory</span>
                        </p>
                      </div>
                    </>
                  )}
                </div>

                <p className="text-secondary" style={{ fontSize: 'var(--font-size-sm)', marginTop: 'var(--space-md)', color: 'var(--color-text-secondary)' }}>
                  Alternatively, you can download models directly through the Ollama app interface or visit the <a href="https://ollama.com/library" target="_blank" rel="noopener noreferrer" style={{ color: 'var(--color-primary)', textDecoration: 'underline' }}>Ollama Library</a> to see all available models.
                </p>
              </div>

              {/* Step 3 */}
              <div style={{ marginBottom: 'var(--space-md)' }}>
                <h4 className="heading-4" style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-sm)', marginBottom: 'var(--space-sm)' }}>
                  <span style={{
                    background: 'var(--color-primary)',
                    color: 'white',
                    borderRadius: '50%',
                    width: '24px',
                    height: '24px',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    fontSize: 'var(--font-size-sm)',
                    fontWeight: 'var(--font-weight-bold)',
                    flexShrink: 0
                  }}>3</span>
                  You're Ready!
                </h4>
                <p className="text-secondary" style={{ fontSize: 'var(--font-size-base)', lineHeight: 1.6 }}>
                  Once your model is downloaded, it will appear in the "Text Model" dropdown below. Select it and start using AI-powered features in Excel!
                </p>
                <p style={{ fontSize: 'var(--font-size-sm)', color: 'var(--color-text-tertiary)', marginTop: 'var(--space-sm)', fontStyle: 'italic' }}>
                  <strong>Tip:</strong> If you don't see your model in the dropdown, make sure your Ollama service is running locally.
                </p>
              </div>

              <div style={{
                background: 'var(--color-primary-light)',
                border: '1px solid var(--color-primary)',
                borderRadius: 'var(--radius-md)',
                padding: 'var(--space-md)',
                marginTop: 'var(--space-lg)'
              }}>
                <p style={{ fontSize: 'var(--font-size-sm)', color: 'var(--color-text-primary)', margin: 0, lineHeight: 1.5 }}>
                  <strong>💡 Tip:</strong> Not sure which model to choose? Start with <strong>ministral-3:8b</strong>, it's a great all-around model that works well on most computers.
                </p>
              </div>
            </div>
          )}
        </div>

        {/* Main Settings Card */}
        <div className="card" style={{ marginBottom: 'var(--space-xl)' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 'var(--space-xl)', flexWrap: 'wrap', gap: 'var(--space-md)' }}>
            <div>
              <h2 className="heading-2" style={{ margin: 0 }}>Configuration</h2>
            </div>
            <button
              className="btn btn-primary"
              onClick={save}
              disabled={saveStatus === 'saving'}
              style={{ flexShrink: 0 }}
            >
              {saveStatus === 'saving' ? 'Saving...' : saveStatus === 'success' ? '✓ Saved' : 'Save'}
            </button>
          </div>

          <hr className="divider" />

          {/* Model Settings */}
          <div className="section">
            <h3 className="heading-3">Model Settings</h3>

            <div className="input-group">
              <label className="label">Text Model</label>
              {availableModels.length > 0 ? (
                <select
                  value={config.model_name || 'llama3'}
                  onChange={(e) => setConfig({ ...config, model_name: e.target.value })}
                  className="select"
                >
                  {availableModels.map(model => (
                    <option key={model} value={model}>
                      {model} {benchmarkResults[model] ? `(${benchmarkResults[model]} t/s)` : ''}
                    </option>
                  ))}
                </select>
              ) : (
                <input
                  className="input"
                  type="text"
                  value={config.model_name || ''}
                  onChange={(e) => setConfig({ ...config, model_name: e.target.value })}
                  placeholder="e.g., llama3"
                />
              )}
              <p className="input-hint">Ollama model name for text generation</p>

              <button
                onClick={runBenchmark}
                disabled={isBenchmarking || availableModels.length === 0}
                className="btn btn-secondary btn-sm"
                style={{ marginTop: 'var(--space-sm)' }}
              >
                {isBenchmarking ? 'Testing...' : 'Test Speeds'}
              </button>
              {isBenchmarking && (
                <div style={{ fontSize: 'var(--font-size-xs)', color: 'var(--color-text-secondary)', marginTop: 'var(--space-xs)' }}>
                  {benchmarkProgress}
                </div>
              )}
            </div>
          </div>


          <hr className="divider" />

          {/* Formatting Options */}
          <div className="section">
            <h3 className="heading-3">Formatting Options</h3>
            <p className="text-secondary" style={{ fontSize: 'var(--font-size-base)', marginBottom: 'var(--space-lg)' }}>
              Control what information is extracted from Excel and sent to the AI. Disabling these can save context and improve accuracy.
            </p>

            <div className="input-group" style={{ marginBottom: 'var(--space-md)' }}>
              <label className="checkbox-label" style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-sm)', cursor: 'pointer' }}>
                <input
                  type="checkbox"
                  checked={includeStyles}
                  onChange={(e) => setIncludeStyles(e.target.checked)}
                  style={{ width: '16px', height: '16px' }}
                />
                <span style={{ fontSize: 'var(--font-size-base)' }}>Include Text Styles (Bold, Italic)</span>
              </label>
            </div>

            <div className="input-group">
              <label className="checkbox-label" style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-sm)', cursor: 'pointer' }}>
                <input
                  type="checkbox"
                  checked={includeColors}
                  onChange={(e) => setIncludeColors(e.target.checked)}
                  style={{ width: '16px', height: '16px' }}
                />
                <span style={{ fontSize: 'var(--font-size-base)' }}>Include Colors (Background, Font)</span>
              </label>
            </div>

          </div>



          {/* Advanced Settings */}
          <div className="section">
            <h3
              className="heading-3"
              style={{ cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 'var(--space-sm)', userSelect: 'none' }}
              onClick={() => setShowAdvanced(!showAdvanced)}
            >
              <span>{showAdvanced ? '▼' : '▶'}</span> Advanced Settings
            </h3>

            {showAdvanced && (
              <div className="animate-in fade-in slide-in-from-top-2 duration-200">
                <div className="input-group">
                  <label className="label" style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span>Batch Timeout (Minutes)</span>
                    <span className="text-secondary">{Math.round((config.timeout_seconds || 1200) / 60)} min</span>
                  </label>
                  <input
                    type="range"
                    min="1"
                    max="120"
                    step="1"
                    value={Math.round((config.timeout_seconds || 1200) / 60)}
                    onChange={(e) => setConfig({ ...config, timeout_seconds: parseInt(e.target.value) * 60 })}
                    style={{ width: '100%', accentColor: 'var(--color-primary)' }}
                  />
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 'var(--font-size-xs)', color: 'var(--color-text-secondary)', marginTop: 'var(--space-xs)' }}>
                    <span>1 min</span>
                    <span>2 hours</span>
                  </div>
                  <p className="text-secondary" style={{ fontSize: 'var(--font-size-xs)', marginTop: 'var(--space-sm)' }}>
                    Maximum time allowed for a batch job before it is automatically cancelled.
                  </p>
                </div>

                {developerMode && (
                  <>
                    <div className="input-group" style={{ marginTop: 'var(--space-lg)', paddingTop: 'var(--space-lg)', borderTop: '1px solid var(--color-border-light)' }}>
                      <label className="checkbox-label" style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-sm)', cursor: 'pointer' }}>
                        <input
                          type="checkbox"
                          checked={debugPrompts}
                          onChange={(e) => setDebugPrompts(e.target.checked)}
                          style={{ width: '16px', height: '16px' }}
                        />
                        <span style={{ fontSize: 'var(--font-size-base)' }}>Debug Prompts (Write to new sheet)</span>
                      </label>
                      <p className="input-hint" style={{ marginLeft: '24px', marginTop: 'var(--space-xs)' }}>
                        Creates a "Prompt Debug" sheet with the exact data sent to the AI.
                      </p>
                    </div>

                    <div className="input-group" style={{ marginTop: 'var(--space-lg)', paddingTop: 'var(--space-lg)', borderTop: '1px solid var(--color-border-light)' }}>
                      <label className="checkbox-label" style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-sm)', cursor: 'pointer' }}>
                        <input
                          type="checkbox"
                          checked={showInspectButton}
                          onChange={(e) => setShowInspectButton(e.target.checked)}
                          style={{ width: '16px', height: '16px' }}
                        />
                        <span style={{ fontSize: 'var(--font-size-base)' }}>Show Inspect Button</span>
                      </label>
                      <p className="input-hint" style={{ marginLeft: '24px', marginTop: 'var(--space-xs)' }}>
                        Enable the "Inspect" button in the Knowledge Base to view document chunks in Excel.
                      </p>
                    </div>
                  </>
                )}
              </div>
            )}
          </div>


          {/* Status Messages */}
          {saveStatus === 'error' && (
            <div style={{
              padding: 'var(--space-md) var(--space-lg)',
              background: 'var(--color-error-bg)',
              color: 'var(--color-error)',
              borderRadius: 'var(--radius-md)',
              fontSize: 'var(--font-size-base)',
              fontWeight: 'var(--font-weight-medium)',
              display: 'flex',
              alignItems: 'center',
              gap: 'var(--space-sm)'
            }}>
              <span>✕</span>
              <span>Failed to save settings</span>
            </div>
          )}
        </div>



      </div>

    </div >
  );
};

export default App;
