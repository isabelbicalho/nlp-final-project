import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split


def read_input(filename, sep, X_title, y_title):
    corpus = pd.read_csv(filename, sep=sep)
    X = corpus.get(X_title).values
    y = corpus.get(y_title).values
    import pdb; pdb.set_trace()
    for i in range(len(X)-1):
        if type(X[i]) != str or np.isnan(y[i]):
            X = np.delete(X, i)
            y = np.delete(y, i)
    return X, y


def split_input(X, y, test_size):
    return train_test_split(X, y, test_size=test_size, shuffle=False)


def convert_to_one_hot(Y, C):
    # import pdb; pdb.set_trace()
    # Y = np.eye(C)[Y.reshape(-1)]
    # return Y
    cat_sequences = []
    import pdb; pdb.set_trace()
    for i, y in enumerate(Y):
        try:
            cats = np.zeros(C)
            cats[y] = 1.0
            cat_sequences.append(cats)
        except:
            import pdb; pdb.set_trace()
            print (y)
    import pdb; pdb.set_trace()
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
                X_indices[i, j] = word_to_index['-OOV-']
            # Increment j to j + 1
            j = j + 1
            
    return X_indices


def index_map(sentences):
    words = set([])
    for sentence in sentences:
        for word in sentence:
            words.add(word.lower())
    word2index = {word: i + 2 for i, word in enumerate(list(words))}
    word2index['-PAD-'] = 0 # Padding
    word2index['-OOV-'] = 1 # Out of vocabulary
    index2word = {i + 2: word for i, word in enumerate(list(words))}
    index2word[0] = '-PAD-' # Padding
    index2word[1] = '-OOV-' # Out of vocabulary
    return word2index, index2word


def build_model(input_shape, word_to_index):
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
    embedding_layer = Embedding(len(word_to_index), 128)
    
    # Propagate sentence_indices through your embedding layer, you get back the embeddings
    embeddings = embedding_layer(sentence_indices)   
    
    # Propagate the embeddings through an LSTM layer with 128-dimensional hidden state
    # Be careful, the returned output should be a batch of sequences.
    X = LSTM(128, return_sequences=True)(embeddings)
    # Add dropout with a probability of 0.5
    X = Dropout(0.5)(X)
    # Propagate X trough another LSTM layer with 128-dimensional hidden state
    # Be careful, the returned output should be a single hidden state, not a batch of sequences.
    X = LSTM(128)(X)
    # Add dropout with a probability of 0.5
    X = Dropout(0.5)(X)
    # Propagate X through a Dense layer with softmax activation to get back a batch of 5-dimensional vectors.
    X = Dense(5)(X)
    # Add a softmax activation
    X = Activation('softmax')(X)
    
    # Create Model instance which converts sentence_indices into X.
    model = Model(inputs=sentence_indices, outputs=X)
    
    return model
