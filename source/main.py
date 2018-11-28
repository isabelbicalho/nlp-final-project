import numpy as np
import keras
from utils import *

import numpy as np
np.random.seed(1)

X, y = read_input("../data/consumer_reviews_of_amazon_products.csv", sep=",", X_title="reviews.text", y_title="reviews.rating")

X_train, X_test, y_train, y_test = split_input(X, y, test_size=0.2)

y_train = y_train.astype(np.int32) - 1
y_test  = y_test.astype(np.int32) - 1
max_len = len(max(X_train, key=len).split())
max_len = 500
print ("MAX_LENGTH = {}".format(max_len))

word_to_index, index_to_word = index_map(X_train)

X_train_indices = sentences_to_indices(X_train, word_to_index, max_len)

y_oh_train = convert_to_one_hot(y_train, C = 5)
y_oh_test =  convert_to_one_hot(y_test,  C = 5)


print ("Creating model...")
model = build_model((max_len,), word_to_index)
print (model.summary())

print ("Compiling model...")
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', fmeasure, precision, recall])

print ("Training model...")
model.fit(X_train_indices, y_oh_train, epochs = 400, batch_size = 256, shuffle=True, callbacks=[keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.0001, patience=10, verbose=0, mode='min')])
print ("saving...")
model.save('tp.h5')


X_test_indices = sentences_to_indices(X_test, word_to_index, max_len = max_len)
# Y_test_oh = convert_to_one_hot(Y_test, C = 5)

print ("Evaluating test...")
loss, acc = model.evaluate(X_test_indices, y_oh_test)
print()
print("Test accuracy = ", acc)
print("Test loss     = ", loss)


# # This code allows you to see the mislabelled examples
# C = 5
# y_test_oh = np.eye(C)[Y_test.reshape(-1)]
# X_test_indices = sentences_to_indices(X_test, word_to_index, max_len)
# pred = model.predict(X_test_indices)
# for i in range(len(X_test)):
#     x = X_test_indices
#     num = np.argmax(pred[i])
#     if(num != Y_test[i]):
#         print('Expected emoji:'+ label_to_emoji(Y_test[i]) + ' prediction: '+ X_test[i] + label_to_emoji(num).strip())
# 
# 
# # Change the sentence below to see your prediction. Make sure all the words are in the Glove embeddings.  
# x_test = np.array(['i am hungry'])
# X_test_indices = sentences_to_indices(x_test, word_to_index, maxLen)
# print(x_test[0] +' '+  label_to_emoji(np.argmax(model.predict(X_test_indices))))
