import os


script_dir = os.path.dirname(os.path.abspath(__file__))
os.system("cd " + script_dir + "/..")
os.system("docker run -p 8501:8501 streamlit-rulekit-gui")
os.system("start http://localhost:8501")
