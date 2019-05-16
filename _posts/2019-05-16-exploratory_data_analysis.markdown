---
layout: post
title:      "Exploratory Data Analysis ."
date:       2019-05-16 11:28:49 +0000
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

0	7129300520	10/13/2014	221900.0	3	1.00	1180	5650	1.0	NaN	0.0	...	7	1180	0.0	1955	0.0	98178	47.5112	-122.257	1340	5650
1	6414100192	12/9/2014	538000.0	3	2.25	2570	7242	2.0	0.0	0.0	...	7	2170	400.0	1951	1991.0	98125	47.7210	-122.319	1690	7639
2	5631500400	2/25/2015	180000.0	2	1.00	770	10000	1.0	0.0	0.0	...	6	770	0.0	1933	NaN	98028	47.7379	-122.233	2720	8062
3	2487200875	12/9/2014	604000.0	4	3.00	1960	5000	1.0	0.0	0.0	...	7	1050	910.0	1965	0.0	98136	47.5208	-122.393	1360	5000
4	1954400510	2/18/2015	510000.0	3	2.00	1680	8080	1.0	0.0	0.0	...	8	1680	0.0	1987	0.0	98074	47.6168	-122.045	1800	7503
5 rows × 21 columns

df.head() spits out the first 5 rows from the data set, this is to make sure it imported correctly, and to get an initial look at the data we will be working with.  Within the parenthesis you can specify how many lines you wish to see, but left blank, it will default to 5. Along with df.head(), there is df.tail() which (you probably guessed) spits out the last 5 rows unless specified differently. I also called df.shape() which spit out  (21597, 21) , telling me the number of rows and columns. I also used df.sample which is nice because it randomly gives you a sample from the data set, You can run this cell over and over and essentially get a different “snap shot” of what’s inside your data set. 
Now I can start calling on some of the cooler methods, like df.describe(), 
	id	price	bedrooms	bathrooms	sqft_living	sqft_lot	floors	waterfront	view	condition	grade	sqft_above	yr_built	yr_renovated	zipcode	lat	long	sqft_living15	sqft_lot15
count	2.159700e+04	21597.0	21597.0	21597.0	21597.0	21597.0	21597.0	19221.0	21534.0	21597.0	21597.0	21597.0	21597.0	17755.0	21597.0	21597.0	21597.0	21597.0	21597.0
mean	4.580474e+09	540297.0	3.0	2.0	2080.0	15099.0	1.0	0.0	0.0	3.0	8.0	1789.0	1971.0	84.0	98078.0	48.0	-122.0	1987.0	12758.0
std	2.876736e+09	367368.0	1.0	1.0	918.0	41413.0	1.0	0.0	1.0	1.0	1.0	828.0	29.0	400.0	54.0	0.0	0.0	685.0	27274.0
min	1.000102e+06	78000.0	1.0	0.0	370.0	520.0	1.0	0.0	0.0	1.0	3.0	370.0	1900.0	0.0	98001.0	47.0	-123.0	399.0	651.0
25%	2.123049e+09	322000.0	3.0	2.0	1430.0	5040.0	1.0	0.0	0.0	3.0	7.0	1190.0	1951.0	0.0	98033.0	47.0	-122.0	1490.0	5100.0
50%	3.904930e+09	450000.0	3.0	2.0	1910.0	7618.0	2.0	0.0	0.0	3.0	7.0	1560.0	1975.0	0.0	98065.0	48.0	-122.0	1840.0	7620.0
75%	7.308900e+09	645000.0	4.0	2.0	2550.0	10685.0	2.0	0.0	0.0	4.0	8.0	2210.0	1997.0	0.0	98118.0	48.0	-122.0	2360.0	10083.0
max	9.900000e+09	7700000.0	33.0	8.0	13540.0	1651359.0	4.0	1.0	4.0	5.0	13.0	9410.0	2015.0	2015.0	98199.0	48.0	-121.0	6210.0	871200.0
#.describe()basic statistical details like percentile, mean, std etc. of the data

and df.info()
<class 'pandas.core.frame.DataFrame'> RangeIndex: 21597 entries, 0 to 21596 Data columns (total 21 columns): id               21597 non-null int64 date             21597 non-null object price            21597 non-null float64 bedrooms         21597 non-null int64 bathrooms        21597 non-null float64 sqft_living      21597 non-null int64 sqft_lot         21597 non-null int64 floors           21597 non-null float64 waterfront       19221 non-null float64 view             21534 non-null float64 condition        21597 non-null int64 grade            21597 non-null int64 sqft_above       21597 non-null int64 sqft_basement    21597 non-null object yr_built         21597 non-null int64 yr_renovated     17755 non-null float64 zipcode          21597 non-null int64 lat              21597 non-null float64 long             21597 non-null float64 sqft_living15    21597 non-null int64 sqft_lot15       21597 non-null int64 dtypes: float64(8), int64(11), object(2) memory usage: 3.5+ MB

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
If you work in real estate, you now know Tuesday is a day you cant afford to miss. If you are a potential home buyer, maybe you will avoid trying to schedule a bunch of "walkthroughs" on Tuesday, as you are less likely to get a relator’s undivided attention.

	Ultimately, there’s no limit to the number of experiments one can perform in the EDA process – it completely depends on what you’re analyzing, as well as the knowledge of packages such as Pandas and matplotlib. 

