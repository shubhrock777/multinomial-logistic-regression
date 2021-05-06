### Multinomial Regression ####
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

mdata = pd.read_csv("D:/360digitM/assi/module_10/Dataset\mdata.csv")
#2.	Work on each feature of the mdataset to create a mdata dictionary as displayed in the below image
#######feature of the mdataset to create a mdata dictionary

#######feature of the mdataset to create a mdata dictionary



mdata_details =pd.DataFrame({"column name":mdata.columns,
                            "mdata type(in Python)": mdata.dtypes})

            #3.	mdata Pre-mdatacessing
          #3.1 mdata Cleaning, Feature Engineering, etc
          

            

#details of mdata 
mdata.info()
mdata.describe()         

#mdata types        
mdata.dtypes


#checking for na value
mdata.isna().sum()
mdata.isnull().sum()

#checking unique value for each columns
mdata.nunique()

"""	Exploratory Data Analysis (EDA):
	Summary
	Univariate analysis
	Bivariate analysis """
    


EDA ={"column ": mdata.columns,
      "mean": mdata.mean(),
      "median":mdata.median(),
      "mode":mdata.mode(),
      "standard deviation": mdata.std(),
      "variance":mdata.var(),
      "skewness":mdata.skew(),
      "kurtosis":mdata.kurt()}

EDA


# covariance for mdata set 
covariance = mdata.cov()
covariance

# Correlation matrix 
co = mdata.corr()
co


mdata.columns

#removing Unnamed: 0
mdata = mdata.drop("Unnamed: 0", axis = 1)
mdata = mdata.drop("id", axis = 1)
mdata.head(10)

mdata.describe()

mdata.columns


mdata.prog.value_counts()

mdata.honors.value_counts()

mdata.schtyp.value_counts()

mdata.ses.value_counts()

mdata.female.value_counts()

mdata.info()

#dummy variable creation 
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()

mdata['female'] = LE.fit_transform(mdata['female'])
mdata['ses'] = LE.fit_transform(mdata['ses'])
mdata['schtyp'] = LE.fit_transform(mdata['schtyp'])
mdata['honors'] = LE.fit_transform(mdata['honors'])


# Boxplot of independent variable distribution for each category of prog

sns.boxplot(x = "prog", y = "read", data =mdata)
sns.boxplot(x = "prog", y = "write", data = mdata)
sns.boxplot(x = "prog", y = "math", data = mdata)
sns.boxplot(x = "prog", y = "science", data = mdata)

# Scatter plot for each categorical prog of car

sns.stripplot(x = "prog", y = "read", jitter = True, data = mdata)
sns.stripplot(x = "prog", y = "write", jitter = True, data = mdata)
sns.stripplot(x = "prog", y = "math", jitter = True, data =mdata)
sns.stripplot(x = "prog", y = "science", jitter = True, data = mdata)

# Scatter plot between each possible pair of independent variable and also histogram for each independent variable 
sns.pairplot(mdata)
sns.pairplot(mdata, hue = "prog") # With showing the category of each car mdatayn in the scatter plot


#boxplot for every columns
mdata.columns
mdata.nunique()

mdata.boxplot(column=['read', 'write', 'math', 'science'])   #no outlier





# Rearrange the order of the variables
mdata = mdata.iloc[:, [3, 0,1, 2, 4,5,6,7,8]]




def norm_func(i):
    x = (i-i.mean())/(i.std())
    return (x)


mdata1=norm_func(mdata.iloc[:,[4,5,6,7]])
mdata1.describe()


model_df = pd.concat([ mdata.iloc[:,[0,1,2,3,8]],mdata1], axis =1)



"""5.	Model Building
5.1	Build the model on the scaled data (try multiple options)
5.2	Perform Multinomial Regression Modl.
5.3	Train and Test the data and compare accuracies by Confusion Matrix, plot ROC AUC curve.
  5.4 Briefly explain the model output in the documentation.
6.	Share the benefits/impact of the solution - how or in what way the business (client) gets benefit from the solution provided.
"""

train, test = train_test_split(model_df, test_size = 0.2 , random_state = 77)

# ‘multinomial’ option is supported only by the ‘lbfgs’ and ‘newton-cg’ solvers
model = LogisticRegression(multi_class = "multinomial", solver="newton-cg").fit(train.iloc[:, 1:],train.iloc[:, 0])


test_predict = model.predict(test.iloc[:, 1:]) # Test predictions
# Test accuracy 
accuracy_score(test.iloc[:,0], test_predict)

train_predict = model.predict(train.iloc[:, 1:]) # Train predictions 
# Train accuracy 
accuracy_score(train.iloc[:,0], train_predict) 



