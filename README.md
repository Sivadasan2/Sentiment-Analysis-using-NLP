# Sentiment-Analysis-using-NLP

Industry Review:
Hate speech  is  an  unfortunately  common  occurrence  on  the  Internet.  Often social media sites like Facebook and Twitter face the problem of identifying and censoring  problematic  posts  while weighing the right to freedom of speech. The importance of detecting and moderating hate  speech  is  evident  from  the  strong  connection between hate speech and actual hate crimes. Early identification of users promoting  hate  speech  could  enable  outreach  programs that attempt to prevent an escalation from speech to action. Sites such as Twitter and Facebook have been seeking  to  actively  combat  hate  speech. In spite of these reasons, NLP research on hate speech has been very limited, primarily due to the lack of a general definition of hate speech, an analysis of its demographic influences, and an investigation of the most effective features

**Natural Language Processing (NLP):** The discipline of computer science, artificial intelligence and linguistics that is concerned with the creation of computational models that process and understand natural language. These include: making the computer understand the semantic grouping of words (e.g. cat and dog are semantically more similar than cat and spoon), text to speech, language translation and many more

**Sentiment Analysis**: It is the interpretation and classification of emotions (positive, negative and neutral) within text data using text analysis techniques. Sentiment analysis allows organizations to identify public sentiment towards certain words or topics.

In this notebook, we'll develop a Sentiment Analysis model to categorize a tweet as Positive or Negative.

**Dataset and Domain**:
The dataset being used is the sentiment140 dataset. It contains 1,600,000 tweets extracted using the Twitter API. 
The tweets have been annotated (0 = Negative, 1 = Positive), The negative annotates the tweet with hate or has negative text in it. As the dataset is huge, I am unable to add the same here in git.

**Description of the Data Set:**
TextID - The id of the tweet 
text selected - text of tweet
text sentiment- the polarity of the tweet (0 = negative, 1= positive)

**Data Pre-Processing:**

Natural Language Processing, in short NLP, is subfield of Machine learning / AI which deals with linguistics and human languages. NLP deals with interactions between computers and human languages. In other words, it enables and programs computers to understand human languages and process & analyse large amount of natural language data.

Lower Casing: 
Each text is converted to lowercase.

Replacing URLs: 
Links starting with "http" or "https" or "www" are replaced by "URL".

Replacing Emojis:
 Replace emojis by using a pre-defined dictionary containing emojis along with their meaning. (eg: ":)" to "EMOJIsmile")

Replacing Usernames: 
Replace @Usernames with word "USER". (eg: "@Kaggle" to "USER")

Removing Non-Alphabets:
 Replacing characters except Digits and Alphabets with a space.

Removing Consecutive letters:
 3 or more consecutive letters are replaced by 2 letters. (eg: "Heyyyy" to "Heyy")

Removing Short Words:
 Words with length less than 2 are removed.

Removing Stop words: 
Stop words are the English words which does not add much meaning to a sentence. They can safely be ignored without sacrificing the meaning of the sentence. (eg: "the", "he", "have")

Lemmatizing:
 Lemmatization is the process of converting a word to its base form. (e.g: “Great” to “Good”)
 
**Processed Data frame:**
We have attached the additional column for the processed text where the data went through the data preprocessing.
 
![image](https://user-images.githubusercontent.com/76568067/122640303-7f4d6480-d11c-11eb-8f5d-1a9778ad35c3.png)

**Architecture:**

![image](https://user-images.githubusercontent.com/76568067/122640385-db17ed80-d11c-11eb-91c8-20c2f2ca0246.png)


**Feature Engineering:**
Term frequency–inverse document frequency, is a numerical statistic that is intended to reflect how important a word is to a document in a collection or corpus
• Term Frequency – TF

![image](https://user-images.githubusercontent.com/76568067/122640326-a3a94100-d11c-11eb-9eee-4b4b533d7d93.png)

Inverse Document frequency:
IDF is a measure of how important a term is. We need the IDF value because computing just the TF alone is not sufficient to understand the importance of words:

![image](https://user-images.githubusercontent.com/76568067/122640332-a7d55e80-d11c-11eb-9a62-80aa8e1ae4a1.png)

**Project Outcome:**
At the end of the NLP ML modelling, Model can classify the neutral and hate tweets, which will shall help the business to identify the tweet and take some actions.




