

import pickle
import keras.models
from keras import layers
from keras.layers import recurrent
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np
import re



def tokenize(sent):
    
    sent=sent.lower()
    return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]

model=keras.models.load_model('model_10.hdf5') #loading the stored model

word_index_vocab=pickle.load(open("word_index_vocab.pickle","rb")) #loading the stored vocab
ans_dictionary= {value: key for key, value in word_index_vocab.items()}

#print(ans_dictionary)

passage=[]
question=[]
ans=[]
with open('test1.txt') as myfile:
    for lines in myfile:
        #print("")
        lines=lines.strip().split('\t')
        print("lines",lines)
        passage.append(lines[0])
        #print(passage)
        question.append(lines[1])



#tokenizes the passage
tokenizer_passage=[]
for items in passage:
    temp=tokenize(items)
    tokenizer_passage.append(temp)

#print(tokenizer_passage)

#tokenizes the question
tokenizer_question=[]
for items in question:
    temp= tokenize(items)
    tokenizer_question.append(temp)



tokenizer_ans=[]
for items in ans:
    temp=tokenize(items)
    tokenizer_ans.append(temp)



xs=[]
xqs=[]


MAX_SEQUENCE_LENGTH_PASSAGE=550
MAX_SEQUENCE_LENGTH_QUESTION=25
x=[]

#replacing each unique word with unique integer and then padding to max ength
for text in tokenizer_passage:
    for w in text:
        if w in word_index_vocab:
            if w not in word_index_vocab:
                pass
            else:
                x.append(word_index_vocab[w])

    #x = [word_index_vocab[w] for w in text]
    xs.append(x)

#print(xs)
data_passage = pad_sequences(xs, maxlen=MAX_SEQUENCE_LENGTH_PASSAGE)


#replacing each unique word in the passage with a unique integer and then padding with max length
xq=[]
for text in tokenizer_question:
    for w in text:
        if w in word_index_vocab:
            if w not in word_index_vocab:
                pass
            else:
                voc=word_index_vocab[w]
        xq.append(voc)
    #xq= [word_index_vocab[w] for w in text]
    xqs.append(xq)

#print(xqs)

data_question = pad_sequences(xqs, maxlen=MAX_SEQUENCE_LENGTH_QUESTION)



ans=model.predict([data_passage,data_question])


length=np.size(ans,axis=0)
length_1=np.size(ans,axis=1)


#this will predict the answer, the one with maximum probability  will be the answer.
max_indexs=[]
for items in range(length):

    max_val=0
    max_index=-1
    for item in range(length_1):
        if ans[items][item] > max_val:
            max_index = item
            max_val = ans[items][item]

    max_indexs.append(max_index)


for values in max_indexs:
    word=ans_dictionary.get(values)
    print("predicted output ",word) #ideally it should predict the answer over here
                                    #in my model most of the time prints answers as 'the', ',', '.'
