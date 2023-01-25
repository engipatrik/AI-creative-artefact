
# Start by importing the libraries required for the processing
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 

# Reading the raw data as it was downloaded from Kaggle
data=pd.read_csv('heart.csv')

# A quick look at the dimensions of the file
print(f"There are {data.shape[0]} rows in the raw data.") 
print(f"There are {data.shape[1]} columns in the raw data.")

# Let's take a closer look at the columns 
print(data.columns)

# The target variable for these models is the presence of heart disease
data["HeartDisease"].value_counts()

# This is a function to automatically generate labels for pie charts 
def label_function(val):
    return f'{val / 100 * len(data):.0f}\n{val:.0f}%'

# Creating a visualisation to show the class balance 
ax1 = plt.axes()
data.groupby('HeartDisease').size().plot(kind='pie', autopct=label_function, textprops={'fontsize': 20},colors=['tomato', 'gold', 'skyblue'], ax=ax1)
plt.show()


# Now moving on to look at the categorical attiributes present in the dataset
data.info()
print(data.max())
print(data.min())

# Looking at the above we can create lists of the different types of attributes 
categorical_cols = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
numerical_cols = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]
binary_col = ["FastingBS"]

# We want to generate plots to compare how the different categorical attributes
# impact the presence of heart disease
# Loop over the categorical attributes
for each_col in categorical_cols + binary_col:
    # Create plot title 
    plt.title(each_col)
    # Plot as histogram with colouring for heart disease
    sns.histplot(x=each_col, hue="HeartDisease", data=data)
    plt.show()

# Now to first look at the distributions of each numerical datapoint
# Subsetting the columns
num_data = data[numerical_cols]

# Plotting the histograms
num_data.hist(layout=(2,3), figsize=(12,10))

pos_hd_df = data[["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak","HeartDisease"]]
pos_hd_df = pos_hd_df.loc[pos_hd_df['HeartDisease']==1]

neg_hd_df = data[["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak","HeartDisease"]]
neg_hd_df = neg_hd_df.loc[neg_hd_df['HeartDisease']==0]

#Generating a different type of plot for all the different data classes 
for i in numerical_cols:
    plt.figure(figsize=(12,6))
    plt.xlabel(i, fontsize=18)
    sns.kdeplot(data = pos_hd_df[i], fill=True)
    sns.kdeplot(data = neg_hd_df[i], fill=True)


# Processing to get a modelling ready dataset
data["Cholesterol"].value_counts()

data[data.Cholesterol == 0].shape[0]
data[data.RestingBP == 0].shape[0]

# There is clearly some missing data for the cholesterol attribute as this 
# should not be a zero value. 172 values.
avg_chol_value = data.Cholesterol[data.Cholesterol != 0].median()
data.loc[data["Cholesterol"] == 0, "Cholesterol"] = avg_chol_value

avg_rest_value = data.RestingBP[data.RestingBP != 0].median()
data.loc[data["RestingBP"] == 0, "RestingBP"] = avg_rest_value

# Creating a function for one hot encoding, to convert categorical data to 
# numerical values for input into our model 
def OHE(df,dfcolumn):
    dfcolumn.nunique()
    len(df.columns)
    finallencol = (dfcolumn.nunique() - 1) + (len(df.columns)-1)
    dummies = pd.get_dummies(dfcolumn, drop_first=True, prefix=dfcolumn.name)
    df=pd.concat([df,dummies],axis='columns')
    df.drop(columns=dfcolumn.name,axis=1,inplace=True) # We have to drop columns to aviod multi-collinearity
    if(finallencol==len(df.columns)):
      print('One Hot Encoding was successful!') 
      print('')
    else:
      print('Error in OHE XXXX')
    return df

# Running the one hot encoder
data = OHE(data,data['ChestPainType'])
data = OHE(data,data['Sex'])
data = OHE(data,data['RestingECG'])
data = OHE(data,data['ExerciseAngina'])
data = OHE(data,data['ST_Slope'])
      
data.info()

data.to_csv("processed.csv", index=False)

