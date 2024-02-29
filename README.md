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
