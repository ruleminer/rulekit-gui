#!/bin/bash

script_dir=$(dirname "$(realpath "$0")")
cd "$script_dir" || exit

if ! docker images | grep -q "streamlit-rulekit-gui"; then
    docker build -t streamlit-rulekit-gui .
fi

docker run -p 8501:8501 streamlit-rulekit-gui &
xdg-open "http://localhost:8501"
