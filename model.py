'''
author - Saurabh parekh (sbp4709)
this file basically preprocess the input data and trains the model.

'''

import numpy as np

from keras import layers
from keras.layers import recurrent
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
import pickle

import re






def tokenize(sent):
    '''Return the tokens of a sentence including punctuation.

    '''
    sent=sent.lower()
    return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]

full=[]
passage=[]
question=[]
ans=[]

vocab=[]
with open("another_small_file.txt",encoding="utf-8") as myfile:
    for lines in myfile:
        #full=full + lines

        lines=lines.strip().split("\t")
        vocab.append(lines[0])
        vocab.append(lines[1])
        vocab.append(lines[2])
        passage.append(lines[0])
        question.append(lines[1])
        ans.append(lines[2])




        #print(full)


#fulltext= sorted(fulltext)


MAX_SEQUENCE_LENGTH_PASSAGE = 550
MAX_SEQUENCE_LENGTH_QUESTION=25
MAX_SEQUENCE_LENGTH_ANswer=20
MAX_NB_WORDS = 90000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2

vocab_size = len(vocab) + 1




new_vocab=[]
#print(vocab)

for items in vocab:
    '''
    tokenizing all the three passage, question and answer
    '''

    item=tokenize(items)
    new_vocab.append(item)
#print(new_vocab)

word_index_vocab={}

#creating a word to index dictionary
for items in new_vocab:
    for item in items:
    #print(items)
        if word_index_vocab.get(item)==None:
            word_index_vocab[item]=len(word_index_vocab)


#reverse dictionary for predicting the answer
reverse_dictionary= {value: key for key, value in word_index_vocab.items()}



#care should be taken that not same dictionary should be used for the training and predicting
pickle_out=  open("word_index_vocab1.pickle","wb")
pickle.dump(word_index_vocab,pickle_out)


#tokenizing the passage
tokenizer_passage=[]
for items in passage:
    temp=tokenize(items)
    tokenizer_passage.append(temp)

print(tokenizer_passage)

#tokenizing the questions
tokenizer_question=[]
for items in question:
    temp= tokenize(items)
    tokenizer_question.append(temp)

tokenizer_ans=[]

#tokenizing the answers
for items in ans:
    temp=tokenize(items)
    tokenizer_ans.append(temp)



xs=[]
xqs=[]

#creating number sequence of the question
for text in tokenizer_passage:
    x = [word_index_vocab[w] for w in text]
    xs.append(x)

#print(xs)
data_passage = pad_sequences(xs, maxlen=MAX_SEQUENCE_LENGTH_PASSAGE)


#creating the number sequence of the question
for text in tokenizer_question:
    xq= [word_index_vocab[w] for w in text]
    xqs.append(xq)



data_question = pad_sequences(xqs, maxlen=MAX_SEQUENCE_LENGTH_QUESTION)

# creating the answer array mutlihot encoding
counter=0
final_ans=[]
for answers in tokenizer_ans:

    counter=counter+1
    print(counter)
    y = np.zeros(len(word_index_vocab) + 1)
    for item in answers:

        y[word_index_vocab[item]] = 1
    final_ans.append(y)


pred= np.array(final_ans)
#print("force",final_ans)



RNN = recurrent.LSTM
EMBED_HIDDEN_SIZE = 100
SENT_HIDDEN_SIZE = 100
QUERY_HIDDEN_SIZE = 100
BATCH_SIZE = 32
EPOCHS = 100


#encoding the passage sentence
sentence_passage = layers.Input(shape=(MAX_SEQUENCE_LENGTH_PASSAGE,), dtype='int32')
encoded_sentence_passage = layers.Embedding(len(word_index_vocab)+1, EMBED_HIDDEN_SIZE)(sentence_passage)

#encoding the squestion
question = layers.Input(shape=(MAX_SEQUENCE_LENGTH_QUESTION,), dtype='int32')
encoded_question = layers.Embedding(len(word_index_vocab)+1, EMBED_HIDDEN_SIZE)(question)
encoded_question = RNN(EMBED_HIDDEN_SIZE)(encoded_question)
encoded_question = layers.RepeatVector(MAX_SEQUENCE_LENGTH_PASSAGE)(encoded_question)


#merging the layer
merge = layers.add([encoded_sentence_passage, encoded_question])
merge = RNN(EMBED_HIDDEN_SIZE,go_backwards=True)(merge)
predicted = layers.Dense(len(word_index_vocab)+1, activation='softmax')(merge)


model = Model([sentence_passage, question], predicted)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              )




print('Training')
model.fit( [data_passage, data_question], pred,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS
           )


model.save("model_10.hdf5")