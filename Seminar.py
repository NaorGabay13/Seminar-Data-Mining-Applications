#Seminar
import pandas as pd
# We will load the library in order to load a csv file
import numpy as np
# Loading the library in order to use the functions
import matplotlib.pyplot as plt
# Display of graphs
from sklearn.metrics import plot_confusion_matrix
# Presentation of the confusion matrix
from sklearn.model_selection import train_test_split
# Splitting the data into a training group and a control group
from sklearn.preprocessing import minmax_scale
# Normalize the data according to MinMax_Scale


Original_Data = pd.read_csv("DataSetNewCSV.csv")
# The original data is unchanged
DataSet = pd.read_csv("DataSetNewCSV.csv")
# Data set containing 300 rows and 9 columns
DataSet_Convert_Gander = pd.factorize(DataSet.gander)[0]
# Conversion of categorical values ​​from the gander column to 0 and 1 in order to perform normalization
DataSet ['gander'] = DataSet_Convert_Gander
# Replacement of the original column with the new column we created
DataSet_MinMaxScale = DataSet.iloc[:,range(1,9)]
# Preparing the data in order to normalize using only the relevant columns
# Use of all columns except the classification column
Data_Scaled = minmax_scale(DataSet_MinMaxScale ,copy = True)
# Normalize the data to 0 and 1 according to the MinMax_Scale method
DataSet_target = DataSet.iloc[:,0]
# Retrieving the classification column of each point in the data
DataSet_target_Convert_number = pd.factorize(DataSet.Class)[0]
# Conversion of the classification column to numerical values
DataSet ['Class'] = DataSet_target_Convert_number
# Replacement of the original column of the classification with the numerical column we created
trainingSet,testSet,trainingSet_targets,testSet_targets = train_test_split(Data_Scaled,DataSet_target,test_size = 0.2 ,train_size = 0.8)
# Splitting the data into a training group and a control group in order to use classifiers
# 20% is the control group and 80% is the training group


'-----------------------------------------------------'
#Knn algorithm

from sklearn.neighbors import KNeighborsClassifier

KNNClassifier = KNeighborsClassifier(3)
KNNClassifier.fit(trainingSet, trainingSet_targets)
KNNResults = KNNClassifier.predict(testSet)
print("The results of the Knn classifier for the test group are:\n\n",KNNResults)

print('\n')

from sklearn import metrics
print("metrics of Knn classifier:\n",metrics.classification_report(testSet_targets, KNNResults))
# Presentation of the quality indicators of the classifier

plot_confusion_matrix(KNNClassifier,testSet,testSet_targets)
plt.title('Knn Classifier')
plt.show()
#ConfusiomMatrixShow


print('\n')

baby = np.array(["The baby's condition is healthy and developing properly well done!" ,
                 "The cognitive problem can be solved by a number of ways:\n 1. Experience through the senses\n 2. Motional thinking\n 3. Exploration and operation of objects",
                 "The physiological problem can be solved by a number of ways:\n 1. Hold the baby head.\n 2. Lying the baby on his stomach.\n 3. Lying the baby on its back.\n 4. Lying the baby on its side."])
# Display messages to the user in order to improve the existing situation or an update message that the situation is normal
       
i = 0
for i in KNNResults[0:5]:
    if i == 'H':
        print(baby[0],'\n')
    if i == 'C' :
        print(baby[1],'\n')
    if i == 'P' :
        print(baby[2],'\n')
        
print('\n\n')
# A loop in order to display each message accordingly according to the classification result
# Display for 5 classification results to illustrate the idea

'-----------------------------------------------------'

#Naive Bayes algorithm

from sklearn.naive_bayes import GaussianNB
# Loading the desired algorithm

NBClassifier = GaussianNB()
# Using the desired algorithm
NBClassifier.fit(trainingSet, trainingSet_targets)
# The learning process between the training group and classifications
NBResults = NBClassifier.predict(testSet)
# Finding the classification of the new point from the training set
print("The results of the Naive Bayes classifier for the test group are:\n",NBResults)
# Print the result of the classifier

print('\n')

from sklearn import metrics
print("metrics of Naive Bayes classifier:\n",metrics.classification_report(testSet_targets, NBResults))
# Presentation of the quality indicators of the classifier

print('\n')

baby = np.array(["The baby's condition is healthy and developing properly well done!" ,
                 "The cognitive problem can be solved by a number of ways:\n 1. Experience through the senses\n 2. Motional thinking\n 3. Exploration and operation of objects",
                 "The physiological problem can be solved by a number of ways:\n 1. Hold the baby head.\n 2. Lying the baby on his stomach.\n 3. Lying the baby on its back.\n 4. Lying the baby on its side."])
# Display messages to the user in order to improve the existing situation or an update message that the situation is normal
       
i = 0
for i in NBResults[0:5]:
    if i == 'H':
        print(baby[0],'\n')
    if i == 'C' :
        print(baby[1],'\n')
    if i == 'P' :
        print(baby[2],'\n')
        
print('\n\n')
# A loop in order to display each message accordingly according to the classification result
# Display for 5 classification results to illustrate the idea

plot_confusion_matrix(NBClassifier,testSet,testSet_targets)
plt.title('Naive Bayes Classifier')
plt.show()
#ConfusiomMatrixShow



'-------------------------------------------------------'


#Decision Tree algorithm

from sklearn.tree import DecisionTreeClassifier
# Loading the desired algorithm
from sklearn  import tree
# View of the tree
DTClassifier = DecisionTreeClassifier(criterion='gini')
# Using the desired algorithm according to the gini index
DTClassifier.fit(trainingSet, trainingSet_targets)
# The learning process between the training group and classifications
DTResults = DTClassifier.predict(testSet) 
# Finding the classification of the new point from the training set
print("The results of the Decision Tree classifier for the test group are:\n",DTResults)
# Print the result of the classifier

print('\n')

from sklearn import metrics
print("metrics of Decision Tree classifier:\n",metrics.classification_report(testSet_targets, DTResults))
# Presentation of the quality indicators of the classifier

print('\n')

baby = np.array(["The baby's condition is healthy and developing properly well done!" ,
                 "The cognitive problem can be solved by a number of ways:\n 1. Experience through the senses\n 2. Motional thinking\n 3. Exploration and operation of objects",
                 "The physiological problem can be solved by a number of ways:\n 1. Hold the baby head.\n 2. Lying the baby on his stomach.\n 3. Lying the baby on its back.\n 4. Lying the baby on its side."])
# Display messages to the user in order to improve the existing situation or an update message that the situation is normal
       
i = 0
for i in DTResults[0:5]:
    if i == 'H':
        print(baby[0],'\n')
    if i == 'C' :
        print(baby[1],'\n')
    if i == 'P' :
        print(baby[2],'\n')
        
# A loop in order to display each message accordingly according to the classification result
# Display for 5 classification results to illustrate the idea


plot_confusion_matrix(DTClassifier,testSet,testSet_targets)
plt.title('Decision Tree Classifier')
plt.show()
#ConfusiomMatrixShow

plt.figure(figsize = (40,45))
# Layout of the tree display
tree.plot_tree(DTClassifier,max_depth = None, fontsize = 16,filled = True,class_names = ['H','P','C'])
plt.show()
# Graphical presentation of the tree


'---------------------------------------------'

# Presentation of the correlation matrix between each column and column excluding the classification column
correlationMatrix = DataSet_MinMaxScale.corr()
print('\n')
print('The strongest relationship between columns according to the calculation of the correlation is:\nAge and height ')
# Print the two columns with the strongest relationship according to the correlation matrix

# Presentation of the linear regression line in order to illustrate the dependence between the two columns age and height
# You can see from the graph that the relationship is strong and positive
Height = DataSet.iloc[:,5].values
Age = DataSet.iloc[:,7].values

n = len(Height)
sumX = np.sum(Height)
sumY = np.sum(Age)
sumX2 = np.sum(Height**2)
sumXY = np.sum(Height*Age)
sumY2 = np.sum(Age**2)

a = (n*sumXY - sumX*sumY)/(n*sumX2-(sumX**2))
b = (sumY*sumX2-sumX*sumXY)/(n*sumX2-(sumX**2))
# y = ax+b

x_plot = np.linspace(0,200,100)
y_hat = a*x_plot + b
plt.scatter(Height,Age)
plt.plot(x_plot,y_hat)
plt.xlabel('Height')
plt.ylabel('Age')
plt.title('Linear regression')

'-------------------------------------------------------'

# Presentation of the correlation index between each column separately and the classification column
# Checking the dependency between each column and the classification column
Data_Corr = DataSet.corr()
correlationMatrix_Classfier = Data_Corr.iloc[:,0]

'----------------------------------------------------------'

# Graphical presentation of all the babies and their health status
Healthy = np.sum(Original_Data.iloc[:,0] == 'H')
CognitiveProblem = np.sum(Original_Data.iloc[:,0] == 'C')
PhysiologicalProblem =  np.sum(Original_Data.iloc[:,0] == 'P')
df = pd.DataFrame({'':['Healthy', 'CognitiveProblem' ,'PhysiologicalProblem'],'Quantity': [Healthy, CognitiveProblem ,PhysiologicalProblem]})
Total_Babies_HealthCondition = df.plot.bar(x = '' ,y = 'Quantity' , rot = 0)
plt.ylim([0, 150])
plt.title('Developmental status')


# Presentation of a pie diagram of the total number of babies in percentages by gender
Male = np.sum(Original_Data.iloc[:,8] == 'Male')
Female = np.sum(Original_Data.iloc[:,8] == 'Female')
df1 = pd.DataFrame({'':[Male, Female]}, index = ['Male', 'Female'])
plot_Gander_Pie = df1.plot(kind='pie', subplots=True, shadow = True,startangle=90,figsize=(5,5), autopct='%1.1f%%')
plt.title('Gander')
  