---
layout: post
title:      "Module 2: Final Project - Northwind Database"
date:       2019-06-13 22:55:39 +0000
permalink:  module_2_final_project_-_northwind_database
---


## June 13, 2019

   Much like this blog post, I had a hard time figuring out how I wanted to approach this project. It began with  being presented the Northwind Database. We were told that we would need to create 4 seperate scientific hypothesis from the data we were given, and were provided with the first question to ask. the first question asking whether or not discounting products has a statistically significant effect on the quantity of a product in an order. The other 3 (or more) questions were on us though. The first step for me however, was to get a better feel of the database we would be working with. That was my workflow, so that is how I will begin. The Northwind Database is a sample database, about a fictitious company named "Northwind Traders". It captures all the sales transactions that occurs between the company and its customers. It also contains the purchase transactions between Northwind and its suppliers. Here is a better breakdown:
	 
	 
1.  Suppliers and Vendors of Northwind.
2.  The Customers of Northwind.
3.  Details of Northwind traders employees.
4.  Information/details about the products being sold.
5.  Invoices.
6.  Information about the shippers for teh company.
7.  Information on purchase order transactions.
8.  All sales Order transactions.
9.  All Inventory transactions.
10.  Inventory information about current inventory. 


### Great visualization!

![](https://theaccessbuddy.files.wordpress.com/2011/07/image.png?w=429&h=326)

(The Access Buddy, 2011)

   As you can see, there is a lot of information contained within this database. The good news is all the tables have been named well, so you shouldn't have too much difficulty finding what you are looking for. My hope is that anyone who should happen to read this before they start their EDA process (exploratory data analysis), will have a stronger "functional view" of the data base before getting too far "into the weeds". If you can somewhat visualize the business process as a whole, the rest of the process should hopefully be a bit more intuitive. The reason I am stressing this part so much is because if anyone out there is even remotely like me, then by this point in the program you could very well feel yourself burning out. Leading up to this project we were rapidly bombarded with a miraid of concepts. Somewhere between SQL queries, tables, foreign keys, normalization (in it's many, many forms), and more statistics and calculus then I would wish on my worst enemy. One could be forgiven for seeing this new "dummy database" such as Northwind, and simply trying to ignore what the database represents, and just jumping in  and attempting to  start working on it. 
	 I understand this empulse better than anyone. When I first started this project, I immediately connected to the database: 
	
```
# Connecting to database
conn = sqlite3.connect('Northwind_small.sqlite')
c = conn.cursor()
```

I quickly got all my table names printed out:

```
# List of all tables
tables = c.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
tables = [i[0] for i in tables]
print(tables)
```

   Grabbed the Order details table, because that had Discounts, which is what I needed to begin tackling the first (provided) question. I converted that table to a dataframe (don't worry, I'll share a great bit of code for that too), and went about trying to find out how many discounts were given, and how I would contrast those against the total order quantity. I have decided to focus on getting familiar with the dataframe as opposed to working through my particular questions, as there are already plenty of student blogs that do just that for the mod 2 final project. The point is, by the time I had answered that question, I had wasted a lot of time researching how to answer this one question, without ever gaining a better understanding of what I was working on. I had my answer, presumably... But I barely understood what it really meant, or how it affected ...well, anything. I was unsure of what I had done thus far, and I still had the bulk of my project to do still, with no real idea of where to go next. I was frustrated, mostly because I knew this was a poor aproach before hand, but I just wanted to get started and get through the project. So I was paying for my initial hastiness, but do I double down on my mistake, or do I go back, and really take the time to figure out what I was even working with? 
	 I already knew what I needed to do, so after a little bit of sole searching (shameless brouser searches for how to actually break down and view, and interpret the database), I came up with a new strategy. I would take the time to go back, and properly explore the entire Northwind database. That is because of all the suggestions a ran across to help me get through this, one unavoidable truth kept creeping back up. Knowing the business process i.e. understanding what the data is and to which business process does it belong to, will help the "user" to think in the right direction & comprehend the results better. If you stuck it out through this whole blog, then you have already demonstrated that you are most likely much more organized and better prepared for the work ahead then I was before starting, and for that, you have earned a prize! as I promised earlier, I will also share my code for converting a table to a Dataframe, which is at this point in my bootcamp, the only real way I know how to do EDA on these types of dataframes. However, about halfway through my painstaking process of copy and pasting the code for this, I found what is with out a doubt my favorite bit of code for this particular notebook. This loop will convert all of your tables into Dataframes, while pulling all column information as well, renaming and keeping track of those names! you earned it. Now get out there and have some fun!
	 
```
# Loop to put all tables into pandas dataframes
dfs = []
for i in tables:
    table = c.execute('select * from "'+i+'"').fetchall()
    columns = c.execute('PRAGMA table_info("'+i+'")').fetchall()
    df = pd.DataFrame(table, columns=[i[1] for i in columns])
    # function to make a string into variable name
    # great bit of code I found while researching
    conv = i+"_df"
    exec(conv + " = df") # => TableName_df
    # Keep all dataframe names in the list to remember what we have
    dfs.append(conv)
    print(conv)
```

enjoy!








