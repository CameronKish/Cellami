#!/bin/bash

echo "🚀 Starting Cellami Build Process..."

# 1. Build Frontend
echo "📦 Building Frontend..."
cd frontend
npm run build
if [ $? -ne 0 ]; then
    echo "❌ Frontend build failed."
    exit 1
fi
cd ..

# 2. Build Backend (PyInstaller)
echo "🐍 Packaging with PyInstaller..."
# Clean previous build artifacts
rm -rf build dist

# 1.5 Generate ICNS file (Mac Requirement)
echo "🖼️  Generating App Icon..."
ICON_SOURCE="assets/Cellami_Desktop.png"
ICON_DEST="assets/Cellami.icns"
ICONSET_DIR="assets/Cellami.iconset"

if [ -f "$ICON_SOURCE" ]; then
    mkdir -p "$ICONSET_DIR"
    # Generate standard sizes
    sips -z 16 16     "$ICON_SOURCE" --out "$ICONSET_DIR/icon_16x16.png" > /dev/null
    sips -z 32 32     "$ICON_SOURCE" --out "$ICONSET_DIR/icon_16x16@2x.png" > /dev/null
    sips -z 32 32     "$ICON_SOURCE" --out "$ICONSET_DIR/icon_32x32.png" > /dev/null
    sips -z 64 64     "$ICON_SOURCE" --out "$ICONSET_DIR/icon_32x32@2x.png" > /dev/null
    sips -z 128 128   "$ICON_SOURCE" --out "$ICONSET_DIR/icon_128x128.png" > /dev/null
    sips -z 256 256   "$ICON_SOURCE" --out "$ICONSET_DIR/icon_128x128@2x.png" > /dev/null
    sips -z 256 256   "$ICON_SOURCE" --out "$ICONSET_DIR/icon_256x256.png" > /dev/null
    sips -z 512 512   "$ICON_SOURCE" --out "$ICONSET_DIR/icon_256x256@2x.png" > /dev/null
    sips -z 512 512   "$ICON_SOURCE" --out "$ICONSET_DIR/icon_512x512.png" > /dev/null
    
    # Convert iconset to icns
    iconutil -c icns "$ICONSET_DIR" -o "$ICON_DEST"
    rm -rf "$ICONSET_DIR"
    echo "✅ Icon generated: $ICON_DEST"
else
    echo "⚠️  Warning: $ICON_SOURCE not found. Skipping icon generation."
fi

# Run PyInstaller
pyinstaller --name "Cellami" \
    --clean \
    --onefile \
    --windowed \
    --icon "assets/Cellami.icns" \
    --collect-all docling \
    --collect-all docling_core \
    --collect-all docling_parse \
    --collect-all docling-ibm-models \
    --collect-all docx \
    --collect-all fastembed \
    --copy-metadata docling \
    --copy-metadata docling-ibm-models \
    --copy-metadata docling-core \
    --copy-metadata docling-parse \
    --copy-metadata fastembed \
    --add-data "frontend/dist:frontend/dist" \
    --add-data "assets:assets" \
    main.py

if [ $? -ne 0 ]; then
    echo "❌ PyInstaller failed."
    exit 1
fi

# 3. Post-Processing (Mac Only)
# Hide from Dock (LSUIElement = true)
INFO_PLIST="dist/Cellami.app/Contents/Info.plist"
if [ -f "$INFO_PLIST" ]; then
    echo "🍎 Configuring Info.plist to hide Dock icon..."
    # Use plutil to insert the key
    plutil -insert LSUIElement -bool true "$INFO_PLIST"
fi

echo "✅ Build Complete!"
cp manifest.prod.xml dist/
echo "📄 Copied manifest.prod.xml to dist/"
echo "🎉 Your app is ready at: dist/Cellami.app"
