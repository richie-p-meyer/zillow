Project goals:
Explore drivers of home value. Create a model to predict home value.

Project description:
Zillow releases data on their home listings and sales. This project takes data from zillow transactions in 2017 and 3 different fips (6037 - Los Angelos County, 6059 - Orange County, 6111 - Ventura County) to attempt to determine what drives home value.

Project planning (lay out your process through the data science pipeline)
1. Review project goals and requirements
2. Create a plan
3. Acquire and clean data
4. Use visualizations and statistical tests to determine drivers
5. Use machine learning to create models
6. Create a presentation
7. Present findings

Initial hypotheses and/or questions you have of the data, ideas
1. Are newer houses worth more?
2. Does increasing the amount of beds and baths per squarefeet change the value?
3. Does increasing lot size increase value?
4. Does the percentage of house sq ft per lot sq ft change the value?

Data dictionary
'bed' - bedroom count
'bath' - bathroom count
'squarefeet' - square ft of house
'lotsquarefeet' - square ft of lot
'yearbuilt' - year house was built
'fips' - county code
'bb_sqft' - (beds+baths)/squarefeet
'hsf_lsf' - house square ft/lot squareft

Instructions or an explanation of how someone else can reproduce your project and findings (What would someone need to be able to recreate your project on their own?)
1. Download files from https://github.com/richie-p-meyer/zillow
2. Update your acquire file with your own server credentials
3. Run the 'Final_Project' notebook

Key findings, recommendations, and takeaways from your project.
House age, more beds and baths per square ft, lot size, and percentage of square ft between the house and lot are all drivers of value. Our model beats the Baseline model by 32% which is significant.

Baseline: 555441.06   Model: 378565.84  Improvement: 176875.22 = 32%

The next variable I would look at to improve the model would be zip codes and neighborhoods.
