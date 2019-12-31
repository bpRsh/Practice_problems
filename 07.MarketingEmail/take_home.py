'''
This script runs the analyses that I developed to answer the email challenge questions. Along with the analysis, I have included comments to help the reader understand my thinking throughout the process. A summary of the analysis is included in the file 'take_home.md'.
'''
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.discrete.discrete_model import Logit
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from scipy.stats import norm, beta
import numpy as np

def load_data(filename):
    '''
    Loads data
    INPUT: CSV filename
    OUTPUT: pandas DataFrame
    '''
    df = pd.read_csv(filename)
    return df

def check_uniqueness(df):
    '''
    Checks to see that email ids are unique in all three tables (this is stated in the prompt, but always good to double check)
    INPUT: pandas DataFrame
    OUTPUT: Bool - True if email_ids are unique, False otherwise
    '''
    return len(df['email_id']) == len(df['email_id'].unique())

def show_histogram(df,column):
    '''
    Plots histogram of numerical data
    INPUT: pandas DataFrame, string - name of column
    OUTPUT: None
    '''
    df[column].hist()
    title = 'Number of emails sent versus ' + column
    plt.title(title)
    plt.ylabel('Number of emails sent')
    plt.xlabel(column)
    plt.show()

def compare_categorical(df,columns):
    '''
    Shows bar chart of categorical data
    INPUT: pandas DataFrame, List - contains names of columns
    OUTPUT: None
    '''
    df_copy = df.copy()
    df_copy['count'] = 1
    for var in columns:
        df_copy.groupby(var).count()['count'].plot(kind = 'bar')
        title = 'Number of emails sent versus ' + var
        plt.xticks(rotation = 'horizontal')
        plt.title(title)
        plt.ylabel('Number of emails sent')
        plt.show()



def percent_converted(df1,df2):
    '''
    Calculates the percentage of emails that were converted
    INPUT: pandas Dataframe, pandas Dataframe
    OUTPUT: Float - percentage of emails that were converted
    '''
    return float(df2.shape[0])/df1.shape[0]

def dummify(df, drop):
    '''
    Dummifies categorical variables
    INPUT: pandas Dataframe with string categorical variables
    OUTPUT: pandas Dataframe with dummifed categorical variables
    '''
    df_out = df.copy()
    variables = ['hour','email_text','email_version','weekday','user_country']
    df_out = pd.get_dummies(df_out,columns = variables, drop_first = drop)
    return df_out

def create_X_y(df1,df2):
    '''
    Creates feature matrix and target variable
    INPUT: pandas DataFrame, pandas DataFrame
    OUTPUT: pandas DataFrame, pandas Series
    '''
    df_merge = pd.merge(df1,df2,on = 'email_id',how = 'left')
    df_merge['link_clicked'] = df_merge['link_clicked'].fillna(0)
    y = df_merge.pop('link_clicked').copy()
    X = df_merge.copy()
    return X,y

def create_Logit(X,y):
    '''
    creates statsmodels logistic regression model with 'linked click ' as target variable
    INPUT: pandas dataframe
    OUTPUT: statsmodels Logistic Regression model
    '''
    X = X.copy()
    X['constant'] = 1
    X.pop('email_id')
    logit = Logit(y,X)
    model = logit.fit(maxiter = 400)
    return model



def z_test_proportions(y1,y2,alpha = .05):
    '''
    Conducts one-tailed Z test to determine if y1 has a greater proportion of 1s than y2, prints results
    INPUT: pandas Series, pandas Series - 1 indicates link was clicked, 0 indicates that it was not, Float - significance level of test
    OUTPUT: none
    '''
    p1 = y1.mean()
    p2 = y2.mean()
    n1 = y1.shape[0]
    n2 = y2.shape[0]
    p_hat = (y1.sum() + y2.sum())/(n1 + n2)
    numerator = p1-p2
    denominator = (p_hat*(1 - p_hat)*(1./n1 + 1./n2))**.5
    z = numerator/denominator
    p_value = (1 - norm.cdf(z))
    if p_value <=alpha:
        print 'p1:', p1, ', p2:', p2 ,'P-value for test:', p_value,'. The first proportion is significantly greater than the second proportion.', '\n'
    else:
        print 'The first proportion is not significantly higher than the second proportion.', '\n'

def bayesian_probability(y1,y2,graph):
    '''
    Calculated probability that y1 has a greater mean than y2, assuming that the underlying probability that generated each series follows a beta distribution with parameters both equal to 1 (uninformed prior), and that the likelihood function is the bimomial distribution
    INPUT: pandas Series, pandas Series - 1 indicated link was clicked, 0 indicates it was not, Bool - whether or not to plot the estimated distributions
    OUTPUT: probability that the strategy that generated y1 performed better than the strategy that generated y2
    '''
    x = np.linspace(0,1,100)

    y1_successes = y1.sum()
    y1_failures = y1.count() - y1_successes
    p1_dist = beta(y1_successes + 1,y1_failures + 1)

    y2_successes = y2.sum()
    y2_failures = y2.count() - y2_successes
    p2_dist = beta(y2_successes + 1,y2_failures + 1)
    p1_pdf = p1_dist.pdf(x)
    if graph:
        line1 = plt.plot(x, p1_pdf, label= 'Targeted subset', lw=2)
        plt.fill_between(x, 0, p1_pdf, alpha=0.2, color=line1[0].get_c())
        p2_pdf = p2_dist.pdf(x)
        line2 = plt.plot(x, p2_pdf, label='Non-targeted subset', lw=2)
        plt.fill_between(x, 0, p2_pdf, alpha=0.2, color=line2[0].get_c())
        plt.ylabel('probability density')
        plt.xlabel('p')
        plt.title('Probability density for the two subsets')
        plt.legend()
        plt.show()

    return np.mean(p1_dist.rvs(size = 1000) > p2_dist.rvs(size = 1000))




def plot_conversion_proportions(df,variable,n):
    '''
    Given a column name, plots the link-clicked proportion grouped by that variable
    INPUT: pandas DataFrame, String - column name, Int - number of subplot, allowing for side by side display
    OUTPUT: none
    '''
    plt.subplot(1,2,n)
    df.groupby(variable)['link_clicked'].mean().plot(kind = 'bar')
    plt.title('Link-clicked rate by ' + variable )
    plt.ylabel('Link-clicked rate')
    plt.xticks(rotation = 'horizontal')

def power_calculation(p1,p2,alpha,power):
    '''
    Given a hypothesized p1 and p2, as well as the desired power of the test, this function outputs the necessary sample size for testing if p1 is greater than p2 (for simplicity I assume that the sample size for each of the arms of the experiment will be equal to one another)
    INPUT: Float - hypothesized proportion of links clicked according to the new method, Float - hypothesized proportion of links clicked according to the random method, Float - alpha level of test, Float - power of test
    OUTPUT: Necessary sample size
    '''
    term1 = p1*(1-p1) + p2*(1-p2)
    term2 = ((norm.ppf(1-alpha) - norm.ppf(1- power))/(p1-p2))**2
    n = term1*term2
    return int(np.ceil(n))





if __name__ == '__main__':

    ### Exploratory Data Analysis ###

    # Load the data
    emails_sent = load_data('email_table.csv')
    emails_opened = load_data('email_opened_table.csv')
    emails_opened['email_opened'] = 1
    links_clicked = load_data('link_clicked_table.csv')
    links_clicked['link_clicked'] = 1

    # Take a quick look at data; looks from this, can see that there is no missing data. There are string categorical variables, though, so we will have to deal with this when we analyze.
    print emails_sent.info(), '\n'
    print emails_opened.info(), '\n'
    print links_clicked.info(), '\n'

    # Check for uniqueness of emails; since all three function calls return true, they are.
    print check_uniqueness(emails_sent), '\n'
    print check_uniqueness(emails_opened), '\n'
    print check_uniqueness(links_clicked), '\n'

    # Take a look at histograms of numeric variables; from this, we note that 'user_past_purchases' is very right skewed.
    show_histogram(emails_sent,'hour')
    show_histogram(emails_sent,'user_past_purchases')

    # Take a look at bar charts of categorical variables; all of these are evently distributed, except for the majority of the emails are sent to people in the United States (over 60%).
    compare_categorical(emails_sent, ['email_text','email_version','weekday','user_country'])

    # Create a DataFrame with dummified categorical variables, dropping the first category to avoid perfect collinearity.
    dummified_df_drop_one = dummify(emails_sent,True)

    # Calculate percentage of emails sent that were opened; answer is 10.3%
    percentage_opened = percent_converted(emails_sent,emails_opened)

    # Calculate percentage of emails sent that resulted in a link clicked; answer is 2.1%
    links_clicked_percentage_sent_email = percent_converted(emails_sent,links_clicked)

    # Calculate percentage of opened emails that resulted in link clicked; answer is 20.5%
    links_clicked_percentage_open_email = percent_converted(emails_opened,links_clicked)











    ### In-depth Analysis ###

    # We want to take a look at what characteristics lead to users clicking on the link. So we need to merge the emails sent table with the links-clicked table.

    # Create a version without dummified categorical variables, used for plotting.
    X_no_dummies, y_no_dummies = create_X_y(emails_sent,links_clicked)
    # create a version with dummified categorical variables, used for model building.
    X,y = create_X_y(dummified_df_drop_one,links_clicked)
    #X.pop('email_id')
    # My idea is to determine what the characteristics associated with higher link-clicked rate are, restrict the test portion of the dataset to emails that had those characteristics, and then calculate the link clicked rate of that subset to estimate how my the results of my model would improve the rate. For interpretability, I choose a logistic regression model. Since we are dealing with unbalanced classes here, I will need to either undersample or oversample to make sure the underrepresented class (here, the users who clicked on the link) is accurately represented. I will use random oversampling method provided by the Python libarary imbalanced-learn.

    # I also create a train-test split here, to avoid overly optimistic estimates of the click through rates. I use a relatively large test size because in the end, when I calculate the link clicked percentage among only those individuals in the test set with the desired characteristics, this subset may be significantly smaller than the full test set. Also note that I use the random oversampling technique AFTER the train_test_split; otherwise test set leakage may occur.

    random_over_sampler = RandomOverSampler()
    X_train, X_test,y_train, y_test = train_test_split(X,y,test_size = .5, random_state = 1)
    X_train_over_sample, y_train_over_sample= random_over_sampler.fit_sample(X_train,y_train)
    X_train_over_sample = pd.DataFrame(X_train_over_sample)
    X_train_over_sample.columns = X_train.columns
    model = create_Logit(X_train_over_sample,y_train_over_sample)
    print model.summary(), '\n'

    # From this, the factors that are associated with higher click through rate are hour of day, past purchases, shorter emails, personalized emails, sending emails Sunday through Monday instead of Friday , and recipient being in the UK and US instead of France or Spain. Looking at the hours of the day, I notice an interesting pattern - hours 10,22,and 23 all have large positive beta coefficients, and they are very significant. This could be because people are getting into the workflow at 10 AM, and are therefore more likely to be checking email. So I will keep emails in the test set during this hours. Also, hours 22 and 23 have very large coefficients, so I will keep these emails as well (maybe people are checking their emails right before going to bed?) Looking at past purchases, since it is continuous, we will want to bin it to figure out above what threshold of previous item purchases to include.

    print X_train['user_past_purchases'].describe(), '\n'
    # The upper quartile is 6 or more items purchased; so we will target those people.

    # So now, we take a look at the held out data, and view only those emails which meet those characteristics, and calculate those click through rates

    # Create dataframe of held-out test data
    df_test = X_test.copy()
    df_test['link_clicked'] = y_test.copy()

    # Create indicator variable for whether or not email meets the desired characteristics
    df_test['subset'] = 'Non-targeted subset'
    df_test['subset'][( (df_test['hour_10'] == 1) | (df_test['hour_22'] == 1) | (df_test['hour_23'] == 1) ) & (df_test['user_past_purchases'] > 6) & (df_test['email_text_short_email'] == 1) &
    (df_test['email_version_personalized'] == 1) & ((df_test['weekday_Wednesday']== 1) | (df_test['weekday_Thursday'] == 1)) & ((df_test['user_country_UK'] == 1)|(df_test['user_country_US'] ==1 ))] = 'Targeted subset'


    # Print the click through rate for the non targeted subset versus the targeted subset.
    print df_test.groupby('subset')['link_clicked'].mean(), '\n'

    # Plot the click through rates by subset.
    df_test.groupby('subset')['link_clicked'].mean().plot(kind = 'bar')
    plt.title('Link-clicked rate by subset')
    plt.ylabel('Link-clicked rate')
    plt.xticks(rotation = 'horizontal')
    plt.show()

    # Conduct Z test for targeted subset having a higher proportion of links clicked than not-targeted subset.
    z_test_proportions(df_test[df_test['subset'] == 'Targeted subset']['link_clicked'],df_test[df_test['subset'] == 'Non-targeted subset']['link_clicked'],.05)

    # Again compare proportions, this time from Bayesian perspective, and visualize estimated distributions
    print 'Bayesian method, probability that first group has better success rate than second group:', bayesian_probability(df_test[df_test['subset'] == 'Targeted subset']['link_clicked'],df_test[df_test['subset'] == 'Non-targeted subset']['link_clicked'],True), '\n'




    # From this test we see that the restricted subset has a much higher proportion of links clicked than the rest of the emails: specifically 23% versus 2%. To test this, we could set up a trial where emails are sent to customers meeting the above characteristics, with the emails themselves also meeting the above characteristics, and then compare this to the previous random method. There are several methods for doing this sort of A/B testing: we could send out an equal number of emails according to the random method and according to my specifications, and then compare the proportions of link-clicked between the two groups either using frequentist methods (Z test for proportions) or Bayesian methods (utilizing a beta prior and posterior). Alternatively, I could set up a multi-armed bandit strategy so that I could simultaneously compare the two methods while taking advantage of the one that performs better. For the Z-test approach, I have written a function to compute the necessary sample size (number of emails sent out) for a given power. This function is called "power_calculation".

    # To compute sample size, I assume that the results of the future test will be similar to the results in this test set, i.e. 22.5% versus 2.1%.
    print 'For power of .8 and alpha .05, sample size for each group should be:',power_calculation(.225,.021,.05,.8), '\n'
    print 'For power of .9 and alpha .05, sample size for each group should be:',power_calculation(.225,.021,.05,.9), '\n'
    print 'For power of .99 and alpha .01, sample size for each group should be:',power_calculation(.225,.021,.01,.99), '\n'

    # Once the data has been gathered, the function "Z_test_proportions" can be used to analyze the difference between the two groups, or the function ""bayesian_probability" can be used if a Bayesian approach is desired.



    # To answer the final question about performance of the campaign among different segments of users, I note that there are only two fields of information we have on users: country and past purchases. The results of the logistic regression indicate that the US and UK are associated with higher click through rates, and that users with more past purchases are more likely to follow the link. The following barplots show this.


    X_no_dummies['link_clicked'] = y_no_dummies

    # Create quartile categories for previous items purchased.
    X_no_dummies['quartile_purchased_items'] = pd.cut(X_no_dummies['user_past_purchases'],[0,1,3,6,21],include_lowest=True)

    # Plot the bar chart by country.
    plot_conversion_proportions(X_no_dummies,'user_country',1)

    # Test for significance of proportion difference of US/UK users versus France/Spain users; note that since I am doing two tests here, both on country and items purchased quartile, I divide the traditional .05 alpha level by 2 (Bonferronni correction for multiple testing).
    FR_ES_users = X_no_dummies[X_no_dummies['user_country'].isin(['FR','ES'])]
    US_UK_users = X_no_dummies[X_no_dummies['user_country'].isin(['UK','US'])]
    z_test_proportions(US_UK_users['link_clicked'],FR_ES_users['link_clicked'],.025)

    # Again compare proportions, this time from Bayesian perspective
    print 'Bayesian method, probability that first group has better success rate than second group:', bayesian_probability(US_UK_users['link_clicked'],FR_ES_users['link_clicked'],False), '\n'

    # Plot the bar chart by past purchases quartile.
    plot_conversion_proportions(X_no_dummies,'quartile_purchased_items',2)
    plt.show()

    # Test for significance of proportion difference of upper quartile of previous items purchased versus lower quartiles.
    first_three_quartiles = X_no_dummies[X_no_dummies['quartile_purchased_items'].isin(['(0, 1]','(1, 3]','(3, 6]'])]
    upper_quartile = X_no_dummies[X_no_dummies['quartile_purchased_items'] == '(6, 21]']
    z_test_proportions(upper_quartile['link_clicked'], first_three_quartiles['link_clicked'],.025)

    # Again compare proportions, this time from Bayesian perspective
    print 'Bayesian method, probability that first group has better success rate than second group:', bayesian_probability(upper_quartile['link_clicked'], first_three_quartiles['link_clicked'],False), '\n'
