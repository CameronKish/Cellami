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

    // Monitor Lifecycle
    if (swr.installing) {
      logToScreen("SW: State = Installing...");
    }
    if (swr.waiting) {
      logToScreen("SW: State = Waiting...");
    }
    if (swr.active) {
      logToScreen("SW: State = Active!");
    }
  },
  onRegisterError(error) {
    logToScreen(`SW: Error ❌ ${error}`);
  }
})

// Listen for Controller Change (The moment it takes over)
navigator.serviceWorker.addEventListener('controllerchange', () => {
  logToScreen("SW: CONTROLLER CHANGED! (Page is now controlled) 🎮");
});

// Check current status
navigator.serviceWorker.ready.then(async (reg) => {
  logToScreen(`SW: Ready Check (Active: ${!!reg.active})`);

  // AUDIT: List EVERYTHING in the cache
  try {
    const cacheNames = await caches.keys();
    logToScreen(`Caches found: ${cacheNames.join(', ')}`);

    for (const name of cacheNames) {
      const cache = await caches.open(name);
      const keys = await cache.keys();
      const urls = keys.map(k => {
        const url = new URL(k.url);
        return url.pathname;
      });
      logToScreen(`[${name}] Contains: ${urls.join(', ')}`);
    }
  } catch (e) {
    logToScreen(`Audit Error: ${e}`);
  }

  // FORCE CACHE: Manually stuff index.html into the pocket
  try {
    const cache = await caches.open('pages');
    await cache.add('/index.html');
    logToScreen("FORCE CACHE: Manually added /index.html to 'pages' ✅");

    // Also add root just in case
    await cache.add('/');
    logToScreen("FORCE CACHE: Manually added / to 'pages' ✅");
  } catch (e) {
    logToScreen(`FORCE CACHE FAILED: ${e}`);
  }

  if (navigator.serviceWorker.controller) {
    logToScreen("SW: Page is ALREADY Controlled. ✅");
  } else {
    logToScreen("SW: Page is NOT controlled yet. ⏳");
  }
});

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
