import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.jsx'
import { registerSW } from 'virtual:pwa-register'

// Debug Logger Helper
const logToScreen = (msg) => {
  const el = document.getElementById('pwa-debug');
  if (el) {
    el.innerHTML += `<div>${new Date().toLocaleTimeString()} ${msg}</div>`;
  } else {
    console.log(msg);
  }
};

const updateSW = registerSW({
  onNeedRefresh() {
    logToScreen("SW: Need Refresh (Update available)");
  },
  onOfflineReady() {
    logToScreen("SW: Offline Ready! ✅");
  },
  onRegistered(swr) {
    logToScreen(`SW: Registered! Scope: ${swr.scope}`);
  },
  onRegisterError(error) {
    logToScreen(`SW: Error ❌ ${error}`);
  }
})

// Create Debug Overlay
const debugDiv = document.createElement('div');
debugDiv.id = 'pwa-debug';
debugDiv.style.cssText = `
  position: fixed; bottom: 0; left: 0; right: 0;
  background: rgba(0,0,0,0.8); color: lime; 
  font-family: monospace; font-size: 10px;
  max-height: 100px; overflow: auto; z-index: 9999;
  padding: 5px; pointer-events: none;
`;
document.body.appendChild(debugDiv);
logToScreen("App Started. Initializing SW...");



createRoot(document.getElementById('root')).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
