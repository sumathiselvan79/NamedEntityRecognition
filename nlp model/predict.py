
# coding: utf-8

# In[20]:


#saving the models
# serialize model to JSON
import nltk
from keras.models import model_from_json
#pading the sequence
from keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np 

# later...
 
# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
f = open('storeTag.pckl', 'rb')
word2idx = pickle.load(f)
f.close()
f = open('store.pckl', 'rb')
tag2idx = pickle.load(f)
f.close()
f = open('tags.pckl', 'rb')
tags = pickle.load(f)
f.close()


# In[63]:


def predict(test_sentence=""):
    test_sentence=  nltk.word_tokenize(test_sentence)
    x_test_sent = pad_sequences(sequences=[[word2idx.get(w, 0) for w in test_sentence]],
                            padding="post", value=0, maxlen=70)
    p = loaded_model.predict(np.array([x_test_sent[0]]))
    p = np.argmax(p, axis=-1)
   # print("{:15}||{}".format("Word", "Prediction"))
   # print(30 * "=")
    array=[]
    for w, pred in zip(test_sentence, p[0]):
    #    print("{:15}: {:5}".format(w, tags[pred]))
        array.append((w,tags[pred]))
    #print(array)
    return_text=""
    for  text,tag in array:
        if tag=="I-geo":
            return_text+=text
    return return_text


# In[64]:


predict("pallikaranai where i live")


# In[61]:


# test_sentence="thilak god is my only one god"
# test_sentence=  nltk.word_tokenize(test_sentence)
# x_test_sent = pad_sequences(sequences=[[word2idx.get(w, 0) for w in test_sentence]],
#                             padding="post", value=0, maxlen=70)
# p = loaded_model.predict(np.array([x_test_sent[0]]))
# p = np.argmax(p, axis=-1)
# print("{:15}||{}".format("Word", "Prediction"))
# print(30 * "=")
# array=[]
# #print(p[0],test_sentence)
# for w, pred in zip(test_sentence, p[0]):
#     print(pred)
#     print("{:15}: {:5}".format(w, tags[pred]))
#     array.append((w,tags[pred]))
# array


# In[ ]:


#create an array
#push element 
#with tags

