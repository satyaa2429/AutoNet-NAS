@echo off
setlocal
pushd "%~dp0" || exit /b 1

if not exist ".venv\Scripts\python.exe" (
    echo The AutoNet-NAS Python environment is missing.
    echo Follow the virtual environment setup in UPDATE_README.md.
    popd
    exit /b 1
)

".venv\Scripts\python.exe" -m streamlit run "app\streamlit_app.py" --server.headless false --server.showEmailPrompt false
set "autonet_exit=%errorlevel%"
popd
exit /b %autonet_exit%
