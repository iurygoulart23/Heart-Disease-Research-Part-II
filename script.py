# import libraries
import codecademylib3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns

# load data
heart = pd.read_csv('heart_disease.csv')
print(heart.head())

#2 creating a boxplot of thalach and heart_disease side by side
sns.boxplot(x = heart.heart_disease, y = heart.thalach)
plt.show()
# yes, i think that have a relationship btw theses 2 variables

#3, 4 investigating more btw those relationship
thalach_hd = heart.thalach[heart.heart_disease == 'presence']

thalach_no_hd = heart.thalach[heart.heart_disease == 'absence']

mean_thalach_hd = np.mean(thalach_hd)
mean_thalach_no_hd = np.mean(thalach_no_hd)

mean_differences = mean_thalach_no_hd - mean_thalach_hd

print('Mean difference for thalach hd and no thalach hd are:', round(mean_differences, 4), '\n')

median_differences = np.median(thalach_no_hd) - np.median(thalach_hd)

print('Diferrence btw thalach hd median and thalach no hd median:', median_differences, '\n')

#5, 6 inspecting if the average thalach heart disease patient is significantly different from the average thalach for a person without hd 
from scipy.stats import ttest_ind

# null hypothesis: the average thalach for a person with hd = to average thalach for a person without hd
# alternative hypotesis: is not equal
tstat, pval = ttest_ind(thalach_hd, thalach_no_hd)
print('The value of p is:', (pval), '\n')
# using a significant treshold of .05 is there a significant difference in average, so reject the null hypothesis

#7 using the same process to investigate other quantitative variable, i choose cholesterol
plt.clf()
sns.boxplot(x = heart.heart_disease, y = heart.chol)
plt.show()

chol_hd = heart.chol[heart.heart_disease == 'presence']
chol_no_hd = heart.chol[heart.heart_disease == 'absence']

chol_mean_difference = np.mean(chol_no_hd) - np.mean(chol_hd)
print('The mean difference btw chol_hd and chol_no_hd is:', chol_mean_difference, '\n')

chol_median_difference = np.median(chol_no_hd) - np.median(chol_hd)
print('The median difference btw chol_no_hd and chol_hd is:', chol_median_difference, '\n')

tstat, pval = ttest_ind(chol_hd, chol_no_hd)
print('The p value is:', pval, '\n')
# using a significance treshold of .05 there is no significance difference, so i can conclude the average btw these 2 variables are equal

#8 investigating relationship btw thalach and type of heart pain
plt.clf()
sns.boxplot(x = heart.cp, y = heart.thalach)
plt.show()

#9 saving the values of each kind of heart disease
thalach_typical = heart.thalach[heart.cp == 'typical angina']
thalach_asymptom = heart.thalach[heart.cp == 'asymptomatic']
thalach_asymptom = heart.thalach[heart.cp == 'non-anginal pain']
thalach_atypical = heart.thalach[heart.cp == 'atypical angina']

#10 runing a single hypothesis test 
#null: People with typical angina, non-anginal pain, atypical angina, and asymptomatic people all have the same average thalach
#alternative: do not all have the same average+
from scipy.stats import f_oneway
tstat, pval = f_oneway(thalach_typical, thalach_asymptom, thalach_asymptom, thalach_atypical)
print('The p-value of ANOVA each differente kind of hd is:', pval, '\n')
#using a significance threshold of .05 i can conclude that do not all have the same average

#11 investiganting what variable have these average difference
from statsmodels.stats.multicomp import pairwise_tukeyhsd
results = pairwise_tukeyhsd(heart.thalach, heart.cp)
print(results, '\n')
#with tukey i can conclude that asymptomatic is the variable with different average

#12 investigating relationship btw kind of chest pain a person experience and whether or not they have a hd
Xtab = pd.crosstab(heart.cp, heart.heart_disease)
print(Xtab, '\n')

#13 runing a hypothesis test
#null: There is NOT an association between chest pain type and whether or not someone is diagnosed with hd
#althernative: there is an association
from scipy.stats import chi2_contingency

chi2, pval, dof, exp = chi2_contingency(Xtab)
print('p-value for chi-square test: ', pval)
#using a significance threshold of .05 i can reject the null hypothesis
