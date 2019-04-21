from flask import Flask,request,url_for,redirect,render_template
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sklearn
import pygal
import re
import nltk
import pickle
from nltk.stem.porter import PorterStemmer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from googletrans import Translator
import tweepy
import csv
import pygal
app = Flask(__name__)
hashtag = ""
dataset = pd.read_csv("C:\\Users\\DELL\\python_blog\\app\\combined2.csv")
df = pd.DataFrame()


# In[5]:


len(dataset.iloc[:,1].values)


# In[6]:


dataset.drop(dataset.columns[[0]], axis=1, inplace=True)


# In[7]:


dataset


# In[8]:


nltk.download("stopwords")


# In[10]:


ps = PorterStemmer()


# In[11]:


corpus = []


# In[12]:

m = 1


# In[13]:

print("training")
for i in range(0,len(dataset)):
    review = re.sub("[^a-zA-Z]"," ",dataset["Review"][i] )
    review = review.lower()
    review  = review.split()
    a = []
    for word in review:
        if(word in stopwords.words("english")):
            m = 1  
        else:
            s = ps.stem(word)
            a.append(s)
    review = a
    review = ' '.join(review)
    corpus.append(review)


# In[14]:

print("model trained")
for i in range(0,2):
    review = re.sub("[^a-zA-Z]"," ","this is good" )
    review = review.lower()
    review  = review.split()
    a = []
    for word in review:
        if(word in stopwords.words("english")):
            m = 1  
        else:
            s = ps.stem(word)
            a.append(s)
    review = a
    review = ' '.join(review)
    corpus.append(review)


# In[15]:

cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values


# In[16]:


cv = CountVectorizer(max_features = 1500)
x = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:,1].values


# In[17]:


# X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.30, random_state = 0)
X_train = x[:-2]
X_test = x[-2:]
y_train = y
classifier = RandomForestClassifier()
classifier.fit(X_train, y_train)


# Predicting the Test set results
# y_pred = classifier.predict(X_test)

# # Making the Confusion Matrix
# from sklearn.metrics import confusion_matrix
# cm = confusion_matrix(y_test, y_pred)


# In[18]:


classifier.predict(X_test)


# In[19]:


# run from here
corpus = corpus[:-2]


# In[20]:


translator = Translator()


# In[21]:


str1 = "ये गलत है"


# In[22]:


for i in range(0,2):
    review = re.sub("[^a-zA-Z]"," ",translator.translate(str1).text )
    review = review.lower()
    review  = review.split()
    a = []
    for word in review:
        if(word in stopwords.words("english")):
            m = 1  
        else:
            s = ps.stem(word)
            a.append(s)
    review = a
    review = ' '.join(review)
    corpus.append(review)


# In[23]:

cv = CountVectorizer(max_features = 1500)
x = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:,1].values
# X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.30, random_state = 0)
X_train = x[:-2]
X_test = x[-2:]
y_train = y
classifier = RandomForestClassifier()
classifier.fit(X_train, y_train)


# Predicting the Test set results
# y_pred = classifier.predict(X_test)

# # Making the Confusion Matrix
# from sklearn.metrics import confusion_matrix
# cm = confusion_matrix(y_test, y_pred)
classifier.predict(X_test)


# In[24]:


#new code
corpus = corpus[:-2]



@app.route('/home', methods=['GET', 'POST'])
def hello():
	hashtag=request.form.get('hashtag')
	print(hashtag)
	global corpus


	# In[25]:



	# http://stackoverflow.com/a/13752628/6762004
	RE_EMOJI = re.compile('[\U00010000-\U0010ffff]', flags=re.UNICODE)

	def strip_emoji(text):
	    return RE_EMOJI.sub(r'', text)


	# In[41]:

	print("getting the tweets")
	arr = []
	counter = 0
	####input your credentials here
	consumer_key = 'Qv0Kw5qmwlZAqk93p6R2OFI2X'
	consumer_secret = 'uAfuPdO4yCMQ48rlHXznXjjAGIZskHlELInqKtX5dZaQ0AJZPI'
	access_token = '1089453692137455616-6PHQDg2q3Dk6MPOD8PnsUqRTOaQywS'
	access_token_secret = 'jmOnoyjzjyZ0dUqycPcqVBeHqiFRlZNZ6pqMQKVMiCNUl'

	auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
	auth.set_access_token(access_token, access_token_secret)
	api = tweepy.API(auth,wait_on_rate_limit=True)
	#####United Airlines
	# Open/Create a file to append data
	# csvFile = open('ua3.csv', 'a')
	#Use csv Writer
	# csvWriter = csv.writer(csvFile)

	for tweet in tweepy.Cursor(api.search,q='#' + hashtag,count=100,
	                           lang="hi",
	                           since="2017-04-03").items():
	    print (tweet.created_at, tweet.text)
	    counter = counter + 1
	    if(counter == 100):
	        break
	    arr.append(translator.translate(strip_emoji(tweet.text)).text)
	#     csvWriter.writerow([tweet.created_at, translator.translate(strip_emoji(tweet.text)).text.encode("utf-8")])


	# In[42]:


	for i in range(0,len(arr)):
	    review = re.sub("[^a-zA-Z]"," ",arr[i] )
	    review = review.lower()
	    review  = review.split()
	    a = []
	    for word in review:
	        if(word in stopwords.words("english")):
	            m = 1  
	        else:
	            s = ps.stem(word)
	            a.append(s)
	    review = a
	    review =  ' '.join(review)
	    corpus.append(review)


	# In[43]:


	cv = CountVectorizer(max_features = 1500)
	x = cv.fit_transform(corpus).toarray()
	y = dataset.iloc[:,1].values
	# X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.30, random_state = 0)
	X_train = x[:-len(arr)]
	X_test = x[-len(arr):]
	y_train = y
	# Fitting Naive Bayes to the Training set
	classifier = RandomForestClassifier()
	classifier.fit(X_train, y_train)


	# Predicting the Test set results
	# y_pred = classifier.predict(X_test)

	# # Making the Confusion Matrix
	# from sklearn.metrics import confusion_matrix
	# cm = confusion_matrix(y_test, y_pred)
	arr2 = classifier.predict(X_test)
	# print(arr2)


	# In[44]:


	arr2 = arr2.tolist()


	# In[45]:


	count1 = arr2.count(1)
	count_1 = arr2.count(-1)
	count_neu  = arr2.count(0)


	# In[46]:


	print("pos=" + str(count1) + "  neg="+ str(count_1) + "  neu="+str(count_neu))


	# In[47]:


	neg = count_1
	pos = count1
	neu = count_neu


	# In[48]:
	classifier = KNeighborsClassifier(n_neighbors=5,p=2, metric='minkowski')
	classifier.fit(X_train, y_train)
	arr2 = classifier.predict(X_test)
	arr2 = arr2.tolist()
	count1KNN = arr2.count(1)
	count_1KNN = arr2.count(-1)
	count_neuKNN  = arr2.count(0)
	
	classifier = SVC(kernel = "linear",random_state=0)
	classifier.fit(X_train, y_train)
	arr2 = classifier.predict(X_test)
	arr2 = arr2.tolist()
	count1SVC = arr2.count(1)
	count_1SVC = arr2.count(-1)
	count_neuSVC  = arr2.count(0)
	classifier  = GaussianNB()
	classifier.fit(X_train, y_train)
	arr2 = classifier.predict(X_test)
	arr2 = arr2.tolist()
	count1NB = arr2.count(1)
	count_1NB = arr2.count(-1)
	count_neuNB  = arr2.count(0)
	classifier   = DecisionTreeClassifier()
	classifier.fit(X_train, y_train)
	arr2 = classifier.predict(X_test)
	arr2 = arr2.tolist()
	count1DTC = arr2.count(1)
	count_1DTC = arr2.count(-1)
	count_neuDTC  = arr2.count(0)

	corpus = corpus[:-len(arr)]
	hist = pygal.Histogram()
	hist.add('Results',  [(neg, 0, 10), (neu, 10, 20), (pos, 20, 30)])
	graph_data = hist.render()
	

		




	return render_template("home.html", graph_data = graph_data,count1DTC = count1DTC,count_1DTC = count_1DTC,count_neuDTC = count_neuDTC,count1NB = count1NB,count_1NB = count_1NB,count_neuNB = count_neuNB,count1SVC  = count1SVC,count_1SVC = count_1SVC,count_neuSVC = count_neuSVC,count1KNN = count1KNN,count_1KNN = count_1KNN,count_neuKNN = count_neuKNN)
	

	print("hello")
@app.route("/",methods=['GET', 'POST'])
def home():

	if request.method=='GET':
		return render_template('main.html')
	else:
		return redirect("/home")

@app.route("/sentence",methods=['GET', 'POST'])
def sentence():
	global corpus
	
	if request.method=='GET':
		return render_template('sentence.html')
	else:
		str1=request.form.get('sentence')
		for i in range(0,2):
		    review = re.sub("[^a-zA-Z]"," ",str1 )
		    review = review.lower()
		    review  = review.split()
		    a = []
		    for word in review:
		        if(word in stopwords.words("english")):
		            m = 1  
		        else:
		            s = ps.stem(word)
		            a.append(s)
		    review = a
		    review = ' '.join(review)
		    corpus.append(review)


		# In[23]:

		cv = CountVectorizer(max_features = 1500)
		x = cv.fit_transform(corpus).toarray()
		y = dataset.iloc[:,1].values
		# X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.30, random_state = 0)
		X_train = x[:-2]
		X_test = x[-2:]
		y_train = y
		classifier = RandomForestClassifier()
		classifier.fit(X_train, y_train)


		# Predicting the Test set results
		# y_pred = classifier.predict(X_test)

		# # Making the Confusion Matrix
		# from sklearn.metrics import confusion_matrix
		# cm = confusion_matrix(y_test, y_pred)
		a = classifier.predict(X_test)[0]


		# In[24]:


		#new code
		corpus = corpus[:-2]
		return render_template("sentence2.html", data = a)



if __name__ == '__main__':
	app.run()