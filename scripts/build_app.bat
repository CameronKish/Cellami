@echo off
echo 🚀 Starting Cellami Build Process (Windows)...

REM Ensure we are in the project root
pushd %~dp0
cd ..

REM 1. Build Frontend
echo 📦 Building Frontend...
cd frontend
call npm run build
if %errorlevel% neq 0 (
    echo ❌ Frontend build failed.
    popd
    exit /b %errorlevel%
)
cd ..

REM 2. Build Backend (PyInstaller)
echo 🐍 Packaging with PyInstaller...
if exist build rd /s /q build
if exist dist rd /s /q dist

REM 1.5 Generate ICO file (Windows Requirement)
echo 🖼️  Generating App Icon...
set ICON_SOURCE=assets\Cellami_Template.png
set ICON_DEST=assets\Cellami.ico

if exist "%ICON_SOURCE%" (
    .\venv312\Scripts\python.exe -c "from PIL import Image; img = Image.open(r'%ICON_SOURCE%'); img.save(r'%ICON_DEST%')"
    echo ✅ Icon generated: %ICON_DEST%
    set ICON_FLAG=--icon "%ICON_DEST%"
) else (
    echo ⚠️  Warning: %ICON_SOURCE% not found. Skipping icon generation.
    set ICON_FLAG=
)

.\venv312\Scripts\python.exe -m PyInstaller --name "Cellami" ^
    --clean ^
    --onefile ^
    --noconsole ^
    --splash "assets\Cellami_Desktop.png" ^
    %ICON_FLAG% ^
    --add-binary "mkcert.exe;." ^
    --collect-all docling ^
    --collect-all docling_core ^
    --collect-all docling_parse ^
    --collect-all docling-ibm-models ^
    --collect-all docx ^
    --collect-all fastembed ^
    --copy-metadata docling ^
    --copy-metadata docling-ibm-models ^
    --copy-metadata docling-core ^
    --copy-metadata docling-parse ^
    --copy-metadata fastembed ^
    --add-data "frontend/dist;frontend/dist" ^
    --add-data "assets;assets" ^
    main.py

if %errorlevel% neq 0 (
    echo ❌ PyInstaller failed.
    exit /b %errorlevel%
)

echo ✅ Build Complete!
echo 🎉 Your app is ready at: dist\Cellami.exe
