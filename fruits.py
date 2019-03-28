# Assigning features and label variables
length=[400,100,350,150,450,50]
sweet=[0,300,150,150,300,0]
color=[100,100,150,50,50,150]
fruit=['Banana','Banana','Orange','Orange','Others','Others']
# Import LabelEncoder
from sklearn import preprocessing
#creating labelEncoder
le = preprocessing.LabelEncoder()
# Converting string labels into numbers.
fruit_encoded=le.fit_transform(fruit)
print fruit_encoded

print "length:",length
print "sweet:",sweet
print "color:",color
features=zip(length,sweet,color)
print features
#Import Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB

#Create a Gaussian Classifier
model = GaussianNB()
label=fruit
# Train the model using the training sets
model.fit(features,label)

#Predict Output

predicted= model.predict([[450,0,100],[550,0,100],[150,200,100],[150,200,200],[0,150,300],[150,150,200],[150,100,100],[150,100,10],[100,80,100],[100,90,120],[250,100,100],[300,150,250],[100,80,120],[30,40,20],[250,80,180],[330,0,100],[330,99,95],[278,992,300],[100,150,450]]) 
print "Predicted Value:", predicted


