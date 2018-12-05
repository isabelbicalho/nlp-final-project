import pandas as pd
import numpy as np
import re
from keras import backend as K

from sklearn.model_selection import train_test_split
from keras.models import Model, Sequential
from keras.layers import Dense, Input, Dropout, LSTM, Activation, SpatialDropout1D, MaxPooling1D, Flatten, Conv1D
from keras.optimizers import Adam
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.initializers import glorot_uniform
from nltk.corpus import stopwords
import pandas as pd 
import gzip 

def parse(path): 
    g = gzip.open(path, 'rb') 
    for l in g: 
        yield eval(l) 

def getDF(path): 
    i = 0 
    df = {} 
    for d in parse(path): 
        df[i] = d 
        i += 1 
    return pd.DataFrame.from_dict(df, orient='index') 



def read_glove_vecs(glove_file):
    with open(glove_file, 'r') as f:
        words = set()
        word_to_vec_map = {}
        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            words.add(curr_word)
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)
        
        i = 1
        words_to_index = {}
        index_to_words = {}
        for w in sorted(words):
            words_to_index[w] = i
            index_to_words[i] = w
            i = i + 1
    words_to_index['unk'] = 1
    index_to_words[1] = 'unk'
    return words_to_index, index_to_words, word_to_vec_map


def read_input(filename, sep, X_title, y_title):
    #corpus = pd.read_csv(filename, sep=sep)
    #X = corpus.get(X_title).values
    #y = corpus.get(y_title).values
    df = getDF('../data/reviews_Video_Games_5.json.gz')
    X = df.get('reviewText').values
    y = df.get('overall').values
    import pdb; pdb.set_trace()
    i_list = []
    for i in range(len(X)-1):
        if type(X[i]) != str or np.isnan(y[i]):
            i_list.append(i)
    X = np.delete(X, i_list)
    y = np.delete(y, i_list)
    sw = set(stopwords.words('english'))
    l = 0
    for x in X:
        l += len(x)
        x = x.lower()
        x = re.sub(r"[^a-zA-Z0-9\s]", '', x)
        x = ' '.join([w for w in x.split() if w not in sw])
    print ("MEDIA LENGTH")
    print (l/len(X))
    return X, y


def split_input(X, y, test_size):
    return train_test_split(X, y, test_size=test_size, shuffle=True)


def convert_to_one_hot(Y, C):
    # Y = np.eye(C)[Y.reshape(-1)]
    # return Y
    cat_sequences = []
    for i, y in enumerate(Y):
        cats = np.zeros(C)
        cats[y] = 1.0
        cat_sequences.append(cats)
    return np.array(cat_sequences)


def sentences_to_indices(X, word_to_index, max_len):
    """
    Converts an array of sentences (strings) into an array of indices corresponding to words in the sentences.
    The output shape should be such that it can be given to `Embedding()` (described in Figure 4). 
    
    Arguments:
    X -- array of sentences (strings), of shape (m, 1)
    word_to_index -- a dictionary containing the each word mapped to its index
    max_len -- maximum number of words in a sentence. You can assume every sentence in X is no longer than this. 
    
    Returns:
    X_indices -- array of indices corresponding to words in the sentences from X, of shape (m, max_len)
    """
    
    m = X.shape[0]                                   # number of training examples
    
    # Initialize X_indices as a numpy matrix of zeros and the correct shape (≈ 1 line)
    X_indices = np.zeros((m, max_len))
    
    for i in range(m):                               # loop over training examples
        
        # Convert the ith training sentence in lower case and split is into words. You should get a list of words.
        sentence_words = X[i].lower().split()
        
        # Initialize j to 0
        j = 0
        
        # Loop over the words of sentence_words
        for w in sentence_words:
            if j >= max_len:
                break
            # Set the (i,j)th entry of X_indices to the index of the correct word.
            try:
                X_indices[i, j] = word_to_index[w]
            except KeyError:
                X_indices[i, j] = word_to_index['unk']
            # Increment j to j + 1
            j = j + 1
            
    return X_indices


def pretrained_embedding_layer(word_to_vec_map, word_to_index, max_len):
    vocab_len = len(word_to_index) + 1
    emb_dim   = 50
    
    emb_matrix = np.zeros((vocab_len, emb_dim))

    for word, index in word_to_index.items():
        emb_matrix[index, :] = word_to_vec_map[word]

    embedding_layer = Embedding(vocab_len, emb_dim, input_length=max_len, trainable=False)
    embedding_layer.build((None,))
    embedding_layer.set_weights([emb_matrix])
    return embedding_layer


def index_map(sentences):
    words = set([])
    for sentence in sentences:
        for word in sentence:
            words.add(word.lower())
    word2index = {word: i + 2 for i, word in enumerate(list(words))}
    word2index['-PAD-'] = 0 # Padding
    #word2index['-OOV-'] = 1 # Out of vocabulary
    index2word = {i + 2: word for i, word in enumerate(list(words))}
    index2word[0] = '-PAD-' # Padding
    #index2word[1] = '-OOV-' # Out of vocabulary
    return word2index, index2word


def build_model(input_shape, word_to_vec_map, word_to_index):
    """
    Function creating the model's graph.
    
    Arguments:
    input_shape -- shape of the input, usually (max_len,)
    word_to_vec_map -- dictionary mapping every word in a vocabulary into its 50-dimensional vector representation
    word_to_index -- dictionary mapping from words to their indices in the vocabulary (400,001 words)

    Returns:
    model -- a model instance in Keras
    """

    # Define sentence_indices as the input of the graph, it should be of shape input_shape and dtype 'int32' (as it contains indices).
    sentence_indices = Input(input_shape, dtype=np.int32)
    
    # Create the embedding layer
    # embedding_layer = Embedding(len(word_to_index), 32)
    embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)
    
    # Propagate sentence_indices through your embedding layer, you get back the embeddings
    # embeddings = embedding_layer(sentence_indices)   
    # 
    # # Propagate the embeddings through an LSTM layer with 128-dimensional hidden state
    # # Be careful, the returned output should be a batch of sequences.
    # X = LSTM(128, return_sequences=True)(embeddings)
    # # Add dropout with a probability of 0.5
    # # X = Dropout(0.5)(X)
    # X = SpatialDropout1D(0.2)(X)
    # # Propagate X trough another LSTM layer with 128-dimensional hidden state
    # # Be careful, the returned output should be a single hidden state, not a batch of sequences.
    # X = LSTM(128, dropout=0.2, recurrent_dropout=0.2)(X)
    # # Add dropout with a probability of 0.5
    # # X = Dropout(0.5)(X)
    # # Propagate X through a Dense layer with softmax activation to get back a batch of 5-dimensional vectors.
    # X = Dense(5)(X)
    # # Add a softmax activation
    # X = Activation('softmax')(X)
    
    # Create Model instance which converts sentence_indices into X.
    #model = Sequential(inputs=sentence_indices, outputs=X)
    model = Sequential()
    #model.add(sentence_indices)
    model.add(embedding_layer)
    model.add(Dropout(0.3))
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(250, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(5, activation='softmax'))
    # model.add(LSTM(128, return_sequences=True))
    # model.add(SpatialDropout1D(0.2))
    # model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    # model.add(Dense(5))
    # model.add(Activation('softmax'))
    return model


def conv_model(input_shape, word_to_vec_map, word_to_index):
    """
    Function creating the model's graph.
    
    Arguments:
    input_shape -- shape of the input, usually (max_len,)
    word_to_vec_map -- dictionary mapping every word in a vocabulary into its 50-dimensional vector representation
    word_to_index -- dictionary mapping from words to their indices in the vocabulary (400,001 words)

    Returns:
    model -- a model instance in Keras
    """

    # Define sentence_indices as the input of the graph, it should be of shape input_shape and dtype 'int32' (as it contains indices).
    sentence_indices = Input(input_shape, dtype=np.int32)
    
    # Create the embedding layer
    embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index, input_shape[0])
    model = Sequential()
    #model.add(sentence_indices)
    model.add(embedding_layer)
    model.add(Dropout(0.3))
    model.add(Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.3))
    model.add(Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.3))
    model.add(Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(5, activation='softmax'))
    return model


def lstm_model(input_shape, word_to_vec_map, word_to_index):
    """
    Function creating the model's graph.
    
    Arguments:
    input_shape -- shape of the input, usually (max_len,)
    word_to_vec_map -- dictionary mapping every word in a vocabulary into its 50-dimensional vector representation
    word_to_index -- dictionary mapping from words to their indices in the vocabulary (400,001 words)

    Returns:
    model -- a model instance in Keras
    """

    # Define sentence_indices as the input of the graph, it should be of shape input_shape and dtype 'int32' (as it contains indices).
    sentence_indices = Input(input_shape, dtype=np.int32)
    
    # Create the embedding layer
    # embedding_layer = Embedding(len(word_to_index), 32)
    embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)
    
    # Create Model instance which converts sentence_indices into X.
    #model = Sequential(inputs=sentence_indices, outputs=X)
    model = Sequential()
    model.add(embedding_layer)
    model.add(LSTM(200, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(200))
    model.add(Dropout(0.3))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(5))
    model.add(Activation('softmax'))
    return model


def mixed_model(input_shape, word_to_vec_map, word_to_index):
    """
    Function creating the model's graph.
    
    Arguments:
    input_shape -- shape of the input, usually (max_len,)
    word_to_vec_map -- dictionary mapping every word in a vocabulary into its 50-dimensional vector representation
    word_to_index -- dictionary mapping from words to their indices in the vocabulary (400,001 words)

    Returns:
    model -- a model instance in Keras
    """

    # Define sentence_indices as the input of the graph, it should be of shape input_shape and dtype 'int32' (as it contains indices).
    sentence_indices = Input(input_shape, dtype=np.int32)
    
    # Create the embedding layer
    # embedding_layer = Embedding(len(word_to_index), 32)
    embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)
    
    # Create Model instance which converts sentence_indices into X.
    #model = Sequential(inputs=sentence_indices, outputs=X)
    model = Sequential()
    model.add(embedding_layer)
    model.add(Dropout(0.3))
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.3))
    model.add(LSTM(124, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(Flatten())
    #model.add(Dense(250, activation='relu'))
    model.add(Dense(5, activation='softmax'))
    #model.add(Dense(5))
    #model.add(Activation('softmax'))
    return model


def mixed_model2(input_shape, word_to_vec_map, word_to_index):
    """
    Function creating the model's graph.
    
    Arguments:
    input_shape -- shape of the input, usually (max_len,)
    word_to_vec_map -- dictionary mapping every word in a vocabulary into its 50-dimensional vector representation
    word_to_index -- dictionary mapping from words to their indices in the vocabulary (400,001 words)

    Returns:
    model -- a model instance in Keras
    """

    # Define sentence_indices as the input of the graph, it should be of shape input_shape and dtype 'int32' (as it contains indices).
    sentence_indices = Input(input_shape, dtype=np.int32)
    
    # Create the embedding layer
    # embedding_layer = Embedding(len(word_to_index), 32)
    embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)
    
    # Create Model instance which converts sentence_indices into X.
    #model = Sequential(inputs=sentence_indices, outputs=X)
    model = Sequential()
    model.add(embedding_layer)
    model.add(Dropout(0.3))
    model.add(Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.3))
    model.add(LSTM(200, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(5, activation='softmax'))
    #model.add(Dense(5))
    #model.add(Activation('softmax'))
    return model


def precision(y_true, y_pred):
    """Precision metric.
    Only computes a batch-wise average of precision.
    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    """Recall metric.
    Only computes a batch-wise average of recall.
    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def fmeasure(y_true, y_pred):
    beta = 1
    # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score
