# Rulekit GUI

This library provides a Graphical User Interface (GUI) for the RuleKit tool that is used 
to generate rule sets for classification, regression and survival problems based on a provided dataset.
You can find the RuleKit at `https://github.com/adaa-polsl/RuleKit-python.git`.

For the application to work correctly your dataset should comply with the following requirements:
- the supported file format is CSV,
- UTF-8 encoding, the field separator is a comma `,`, while the decimal is a period `.`,
- missing values are represented as an empty character `""`,
- the first line of the loaded file is the column names,
- the decision attribute must be named `target` in classification and regression, 
   while for survival - `survival_time` and `survival_status`.

After loading a dataset, you will see a preview of it with column names and values. 
Then, in the `Model` tab, you can define the parameters of the rule induction and the type of problem it should address.

![model example](https://www.dropbox.com/scl/fi/b5pvo642tn5s28byiam4r/model.png?rlkey=6kp86zd10lpz7vn9vmhq21kb3&dl=0)


You can look at all the available parameters in the documentation [here](https://adaa-polsl.github.io/RuleKit-python/v1.7.6/index.html)


## Installation and Usage
### Clone repository
First create a working directory into where the application should be placed. 
Then call the following commands in Git Bash:

```
cd PATH TO YOUR DIRECTORY
git clone https://github.com/ruleminer/rulekit-gui.git
```

### Docker

To run the library using Docker, make sure you have **Docker** installed on your system. 
If you are a Windows user in the first place you should start **Docker Desktop** which is responsible for starting the engine.
Then execute the following command in your terminal (CMD):

```
powershell -File [Your path to the "script_to_run_app.ps1" file, which is located in the folder where you cloned the repository]
```

This command will execute a PowerShell script in a command line window, which is responsible 
for creating the Docker Image and running the application as a Container. Make sure you specify the correct 
path to the script, including the `.ps1` file extension.


### Running the application manually
If you don't want to use the PowerShell script you can create the Docker Image yourself by staying in Git Bash.
Make sure you're in your working directory, then type the following command at the command line:

```
docker build -t streamlit-rulekit-gui .
```

This will create an Image containing the proper environment to run the application. 
Then simply run the image in the container using the local host - tab following command:

```
docker run -p 8501:8501 streamlit-rulekit-gui
```

After that you should be able to open the applications in your browser at the address: `http://localhost:8501`.
