import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { VitePWA } from 'vite-plugin-pwa'
import { version } from './package.json'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [
    react(),
    VitePWA({
      registerType: 'autoUpdate',
      includeAssets: ['assets/*'],
      manifest: {
        name: 'Cellami AI',
        short_name: 'Cellami',
        description: 'Local AI for Excel',
        theme_color: '#ffffff',
        icons: [
          {
            src: 'assets/icon-80.png',
            sizes: '192x192',
            type: 'image/png'
          },
          {
            src: 'assets/icon-80.png',
            sizes: '512x512',
            type: 'image/png'
          }
        ]
      },
      workbox: {
        clientsClaim: true,
        skipWaiting: true,
        // CRITICAL: Ignore Excel's query params (?_host_Info=...) so cache matches index.html
        ignoreURLParametersMatching: [/.*/],
        // EXCLUDE 'html' from globPatterns to prevent strict hash check failure (Vercel Injection)
        globPatterns: ['**/*.{js,css,ico,png,svg,json}'],
        navigateFallback: '/index.html',
        runtimeCaching: [
          {
            urlPattern: ({ url }) => url.pathname.startsWith('/api'),
            handler: 'NetworkOnly',
            options: {
              backgroundSync: {
                name: 'apiQueue',
                options: {
                  maxRetentionTime: 24 * 60 // Retry for max of 24 Hours (optional)
                }
              }
            }
          },
          {
            // Cache HTML pages (Navigation) with NetworkFirst
            // This allows Vercel to inject scripts/hashes without breaking the Manifest hash check
            urlPattern: ({ request }) => request.mode === 'navigate',
            handler: 'NetworkFirst',
            options: {
              cacheName: 'pages',
              expiration: {
                maxEntries: 10,
                maxAgeSeconds: 24 * 60 * 60 // 24 Hours
              }
            }
          }
        ]
      },
      devOptions: {
        enabled: true
      }
    })
  ],
  define: {
    __APP_VERSION__: JSON.stringify(version),
    __BUILD_TIMESTAMP__: JSON.stringify(new Date().toLocaleString('en-US', { timeZone: 'America/New_York', dateStyle: 'short', timeStyle: 'short' }) + ' EST')
  },
  server: {
    port: 3000,
    https: false,
    proxy: {
      '/api': {
        target: 'https://127.0.0.1:8000',
        changeOrigin: true,
        secure: false,
        timeout: 300000,
        proxyTimeout: 300000
      }
    }
  },
  preview: {
    port: 3000,
    https: false,
    headers: {
      'Access-Control-Allow-Origin': '*'
    }
  }
})
