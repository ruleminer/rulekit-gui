# Rulekit GUI

To run the application, it is suggested to create an environment using conda, 
and then install all the required packages from the `requirements.txt` file. 

If you already have installed Anaconda, in the console release the following commands:


```
   conda create --name myenv python=3.11
   conda activate myenv
   pip install -r requirements.txt
```

Then, in the console, open the folder where the application source code files 
(`rulekit_gui_app.py` and `gui_functions.py`) are located. 
And call the following command:

```
   cd PATH TO YOUR FILE
   streamlit run rulekit_gui_app.py
```


