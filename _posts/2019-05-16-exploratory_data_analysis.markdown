---
layout: post
title:      "Exploratory Data Analysis ."
date:       2019-05-16 07:28:50 -0400
permalink:  exploratory_data_analysis
---


   There is so much more data in our world today. Pretty much every business collects data today in some form or another.  Collecting data is not a new practice by any means. However, new data analysis and visualization programs allow for reaching even better understanding.  Modern data analytics allow businesses to better understand their efficiency and performance, and will ultimately help the business make more informed decisions. One example could be analyzing consumer attributes in order to create targeted ads. Data analysis can be applied to nearly every aspect of business, as long as one understands the tools available.
Exploratory Data Analysis – EDA – is crucial in understanding the what, why, and how of any problem.  Here is the proper definition: exploratory data analysis is an approach to analyzing data sets by summarizing their main characteristics with visualizations. The EDA process is a crucial step prior to building a model in order to unravel various insights that later become important in developing a robust algorithmic model.
Before you can really start exploring your data, you have to scrub/clean it first. These different processes actually go hand in hand, and you will likely jump back and forth between the two as you work through your data set. For My Mod1 project, the first thing I did was import my libraries that would help me import, open and work with the data. I actually imported a lot of libraries because I am building up to regressions, but the following should get you started:

-import pandas as pd
 -import numpy as np
  -import seaborn as sns
   -import matplotlib from matplotlib
    -import pyplot as plt

From there, I imported my data set:
 df = pd.read_csv("kc_house_data.csv")
df.head()



df.head() spits out the first 5 rows from the data set, this is to make sure it imported correctly, and to get an initial look at the data we will be working with.  Within the parenthesis you can specify how many lines you wish to see, but left blank, it will default to 5. Along with df.head(), there is df.tail() which (you probably guessed) spits out the last 5 rows unless specified differently. I also called df.shape() which spit out  (21597, 21) , telling me the number of rows and columns. I also used df.sample which is nice because it randomly gives you a sample from the data set, You can run this cell over and over and essentially get a different “snap shot” of what’s inside your data set. 
Now I can start calling on some of the cooler methods, like df.describe(), 
	
#.describe()basic statistical details like percentile, mean, std etc. of the data

and df.info()


From this point, I started going through and removing null values, converting data types, and generally doing more scrubbing/conditioning, but I want to bring back the focus to EDA.

EDA
At this stage, I have worked through, and cleaned the data fairly well (and changed my mind about what attributes to focus on several times over).
It is time to start asking our data some questions. Not only we are tasked with asking, and hopefully answering 3 questions, but they need to be relevent questions. The example that was given to us as a bad question was: "does sq ft(living space) affect price?" This is not a bad question because it cant be answered clearly (or super easily), but because it doesn't really teach us anything either. We already know sq ft affects housing prices. This was particularly challenging for me, considering I only started "coding" less than a month ago. At this point I stopped and really started "pouring" though the data again, as well as searching the Web for similar research that had been done, and trying to choose the best questions to ask.
Q1: Is there a correlation between sales and day of the week?
Data Handling
In [20]:
 
df['weekday'] = df['date'].dt.dayofweek

df.weekday.hist()
plt.show()





	
 
Conclusion
Tuesday seems to be the day of the week when most homes are bought. It appears to build up slightly from Monday, to Tuesday, then taper slowly down until the weekend where it drops off drastically (no surprise, banks are closed through the weekend).
Recommendation
If you work in real estate, you now know Tuesday is a day you cant afford to miss. If you are a potential homebuyer, maybe you will avoid trying to schedule a bunch of "walkthroughs" on Tuesday, as you are less likely to get a relator’s undivided attention.

Ultimately, there’s no limit to the number of experiments one can perform in the EDA process – it completely depends on what you’re analyzing, as well as the knowledge of packages such as Pandas and matplotlib.  I went on to ask and answer two more questions in this data set. However, If I had more time  I probably could have came up with better questions.  





