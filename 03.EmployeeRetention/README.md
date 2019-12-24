# Exploratory Data Analysis for Employee Retention Dataset

* Employee turn-over is a very costly problem for companies.
* The cost of replacing an employee if often larger than 100K USD, taking into account the time spent to interview and find a replacement, placement fees, sign-on bonuses and the loss of productivity for several months.
* It is only natural then that data science has started being applied to this area.
* Understanding why and when employees are most likely to leave can lead to actions to improve employee retention as well as planning new hiring in advance. This application of DS is sometimes called people analytics or people data science
* We got employee data from a few companies. We have data about all employees who joined from 2011/01/24 to 2015/12/13. For each employee, we also know if they are still at the company as of 2015/12/13 or they have quit.
* Beside that, we have general info about the employee, such as avg salary during her tenure, dept, and yrs of experience.

**Goal:**

In this challenge, you have a data set with info about the employees and have to predict when employees are going to quit by understanding the main drivers of employee churn.
* Assume, for each company, that the headcount starts from zero on 2011/01/23. Estimate employee headcount, for each company, on each day, from 2011/01/24 to 2015/12/13. That is, if by 2012/03/02 2000 people have joined company 1 and 1000 of them have already quit, then company headcount on 2012/03/02 for company 1 would be 1000.
* You should create a table with 3 columns: day, employee_headcount, company_id. What are the main factors that drive employee churn? Do they make sense? Explain your findings.
* If you could add to this data set just one variable that could help explain employee churn, what would that be?

**Data:** (data/employee_retention_data.csv)

Columns:

* employee_id : id of the employee. Unique by employee per company
* company_id : company id.
* dept : employee dept
* seniority : number of yrs of work experience when hired
* salary: avg yearly salary of the employee during her tenure within the company
* join_date: when the employee joined the company, it can only be between 2011/01/24 and 2015/12/13
* quit_date: when the employee left her job (if she is still employed as of 2015/12/13, this field is NA)


**Question 1**

Function that returns a list of the names of categorical variables

* Define a function with name get_categorical_variables
* Pass dataframe as parameter (Read csv file and convert it into pandas dataframe)
* Return list of all categorical fields available.


**Question 2**

Function that returns the list of the names of numeric variables

* Define a function with name get_numerical_variables
* Pass dataframe as parameter (Read csv file and convert it into pandas dataframe)
* Return list of all numerical fields available.


**Question 3**

Function that returns, for numeric variables, mean, median, 25, 50, 75th percentile

* Define a function with name get_numerical_variables_percentile
* Pass dataframe as parameter (Read csv file and convert it into pandas dataframe)
* Return dataframe with following columns:
    * variable name
    * mean
    * median
    * 25th percentile
    * 50th percentile
    * 75th percentile


**Question 4**

For categorical variables, get modes

* Define a function with name get_categorical_variables_modes
* Pass dataframe as parameter (Read csv file and convert it into pandas dataframe)
* Return dict object with following keys:
    * converted
    * country
    * new_user
    * source


**Question 5**

For each column, list the count of missing values

* Define a function with name get_missing_values_count
* Pass dataframe as parameter (Read csv file and convert it into pandas dataframe)
* Return dataframe with following columns:
    * var_name
    * missing_value_count


**Question 6**

Plot histograms using different subplots of all the numerical values in a single plot

* Define a function with name plot_histogram_with_numerical_values
* Pass dataframe and list of columns you want to plot as parameter
* Plot the graph
* Add column names as plot names (In case you dont understand this please connect with instructor)
* Change the histogram colour to yellow
* Fit a normal curve on those histograms (In case you dont understand this please connect with instructor)
