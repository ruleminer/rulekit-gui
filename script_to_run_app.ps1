$script_dir = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location -Path $script_dir

if (-not (docker images | Select-String -Pattern "streamlit-rulekit-gui")) {
    docker build -t streamlit-rulekit-gui .
}
Start-Job -ScriptBlock {docker run -p 8501:8501 streamlit-rulekit-gui}
Start-Process "http://localhost:8501"