
# coding: utf-8

# ### Titanic Python Exercise
#
# Work book for submission to Decoded Data Fellowship to preeict survical outcome based on gender and Pclass

# In[68]:


# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[69]:


# Read dataset
test = pd.read_csv("test.csv")


# In[70]:


# EDA | Generate crosstab to review the dataset
pd.crosstab(test.Sex, test.Pclass, normalize=False, margins=True)


# In[71]:


# add a column and set intial values
test['ModelPrediction'] = 0


# In[72]:


# assign prediction for females = survived, unless travelling in Pclass = 3 | all males under 18 in PClass 1,2
test.loc[test['Sex'] == "female", 'ModelPrediction'] = 1
test.loc[test['Pclass'] == 3, 'ModelPrediction'] = 0
test.loc[((test['Pclass'] == 1) | (test['Pclass'] == 2)) & (test['Sex'] == "male") & (test.Age < 18), 'ModelPrediction'] = 1


# In[73]:


# review results
test.head()


# In[74]:


# calculate predicted survival outcome
sum(test.ModelPrediction == 1) / test.shape[0] #shape counts the number of rows in the dataset


# In[75]:


# generate  bar chart showing predicted survival outcome
test.groupby(['ModelPrediction', 'Sex']).size().unstack().plot(kind='barh', stacked=True)
plt.title('Predicted survival outcome by gender')
plt.show()


# In[76]:


# Create a new data frame for the output
submission = test.filter(['PassengerId', 'ModelPrediction'])
submission.head()


# In[77]:


# Create the csv file output
submission.to_csv('titanic_in_python_submission.csv', index=False)


# End of Notebook
