import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.jsx'
import { registerSW } from 'virtual:pwa-register'

const updateSW = registerSW({
  onNeedRefresh() {
    // Prompt user to refresh if a new version is available (optional)
    console.log("PWA: New content available, click reload to update.")
  },
  onOfflineReady() {
    console.log("PWA: App is ready for offline use.")
  },
})

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
