def correct_justice_spelling (data_df):
    ''' 
    Correct the spelling errors in the justice names.
    
    data_df: pandas dataframe containing the a column 'name' with errant spellings and a column 'last_name' to add the
    correct spelling to.
    
    returns the corrected dataframe
    '''
    data_df.loc[(data_df.last_name == 'van'), 'last_name'] = 'van devanter'
    data_df.loc[(data_df.last_name == "m'lean"), 'last_name'] = 'mclean'
    data_df.loc[(data_df.last_name == 'daniels'), 'last_name'] = 'daniel'
    data_df.dloc[(data_df.last_name == 'duval') | (data_df.last_name == '*duvall'), 'last_name'] = 'duvall'
    data_df.loc[(data_df.last_name == "m'kinley"), 'last_name'] = 'mckinley'
    data_df.loc[(data_df.last_name == 'brandies'), 'last_name'] = 'brandeis'
    data_df.loc[(data_df.last_name == 'homes'), 'last_name'] = 'holmes'
    data_df.loc[(data_df.last_name == 'johnston') | (data_df.last_name == '*johnson'), 'last_name'] = 'johnson'
    data_df.loc[(data_df.last_name == 'connor'), 'last_name'] = "o'connor"
    data_df.loc[(data_df.last_name == 'bruger'), 'last_name'] = 'burger'
    data_df.loc[(data_df.last_name == 'bulter'), 'last_name'] = 'butler'
    data_df.loc[(data_df.last_name == 'wilso'), 'last_name'] = 'wilson'
    data_df.loc[(data_df.last_name == 'thomson'), 'last_name'] = 'thompson'
    data_df.loc[(data_.last_name == 'black;'), 'last_name'] = 'black'
    data_df.loc[(data_df.last_name == 'millier'), 'last_name'] = 'miller'
    data_df.loc[(data_df.last_name == 'field;'), 'last_name'] = 'field'
    data_df.loc[(data_df.last_name == '[[author]]ginsburg[[/author]]'), 'last_name'] = 'ginsburg'
    data_df.loc[(data.last_name == 'mokenna'), 'last_name'] = 'mckenna'
    
    return data_df

# Helper function to return the most frequent item from the regex match list to serve as the topic label for opinion
def most_common(lst):
    '''Helper function to return the most frequent item from a list.
    
    lst: list; can be python list of numpy array
    
    return: the most frequent item to appear in lst
    
    '''
    return max(set(lst), key=lst.count)

# Formating amendment matches from regex pattern search
def format_amendments(data_df):
    ''' 
    Format the returns from the regex expression match for the word preceding the word 'Amendment'
    
    data_df: pandas dataframe containing the column 'amendment'
    
    returns the formatted dataframe
    
    '''
    
    data_df.loc[(data_df.amendment == 'First') | (data_df.amendment == '1st'), 'amendment'] = 1
    data_df.loc[(data_df.amendment == 'Second') | (data_df.amendment == '2nd'), 'amendment'] = 2
    data_df.loc[(data_df.amendment == 'Third') | (data_df.amendment == '3rd'), 'amendment'] = 3
    data_df.loc[(data_df.amendment == 'Fourth') | (data_df.amendment == '4th'), 'amendment'] = 4
    data_df.loc[(data_df.amendment == 'Fifth') | (data_df.amendment == '5th'), 'amendment'] = 5
    data_df.loc[(data_df.amendment == 'Sixth') | (data_df.amendment == '6th'), 'amendment'] = 6
    data_df.loc[(data_df.amendment == 'Seventh') | (data_df.amendment == '7th'), 'amendment'] = 7
    data_df.loc[(data_df.amendment == 'Eighth') | (data_df.amendment == '8th'), 'amendment'] = 8
    data_df.loc[(data_df.amendment == 'Ninth') | (data_df.amendment == '9th'), 'amendment'] = 9
    data_df.loc[(data_df.amendment == 'Tenth') | (data_df.amendment == '10th'), 'amendment'] = 10
    data_df.loc[(data_df.amendment == 'Eleventh') | (data_df.amendment == '11th'), 'amendment'] = 11
    data_df.loc[(data_df.amendment == 'Twelfth') | (data_df.amendment == '12th'), 'amendment'] = 12
    data_df.loc[(data_df.amendment == 'Thirteenth') | (data_df.amendment == '13th'), 'amendment'] = 13
    data_df.loc[(data_df.amendment == 'Fourteenth') | (data_df.amendment == '14th'), 'amendment'] = 14
    data_df.loc[(data_df.amendment == 'Fifteenth') | (data_df.amendment == '15th'), 'amendment'] = 15
    data_df.loc[(data_df.amendment == 'Sixthteenth') | (data_df.amendment == '16th'), 'amendment'] = 16
    data_df.loc[(data_df.amendment == 'Seventeenth') | (data_df.amendment == '17th'), 'amendment'] = 17
    data_df.loc[(data_df.amendment == 'Eighteenth') | (data_df.amendment == '18th'), 'amendment'] = 18
    data_df.loc[(data_df.amendment == 'Nineteenth') | (data_df.amendment == '19th'), 'amendment'] = 19
    data_df.loc[(data_df.amendment == 'Twentieth') | (data_df.amendment == '20'), 'amendment'] = 20
    data_df.loc[(data_df.amendment == 'first') | (data_df.amendment == '21st'), 'amendment'] = 21
    data_df.loc[(data_df.amendment == 'second') | (data_df.amendment == '22nd'), 'amendment'] = 22
    data_df.loc[(data_df.amendment == 'third') | (data_df.amendment == '23rd'), 'amendment'] = 23
    data_df.loc[(data_df.amendment == 'fourth') | (data_df.amendment == '24th'), 'amendment'] = 24
    data_df.loc[(data_df.amendment == 'fifth') | (data_df.amendment == '25th'), 'amendment'] = 25
    data_df.loc[(data_df.amendment == 'sixth') | (data_df.amendment == '26th'), 'amendment'] = 26
    data_df.loc[(data_df.amendment == 'seventh') | (data_df.amendment == '27th'), 'amendment'] = 27
    data_df.loc[(data_df.amendment == 'eighth') | (data_df.amendment == '28th'), 'amendment'] = 28
    
    return data_df


def tokenize_and_stop (text):
    '''Text preprocessing function that takes in a raw text document, tokenizes it, converts all tokens to lowercase,
    and removes stopwords. This function was designed to be used in the form of DataFrame['col_name'].map(function).
    
    text: string, raw text document as an instance in a pandas dataframe
    
    returns the tokenized and stopped text as a list of tokens
    
    '''
    
    # Use nltk word_tokenize function on the input text
    data_text = word_tokenize(text)
    
    # Convert all tokens to lowercase and remove stopwords
    data_stopped = [tok.lower() for tok in data_text if tok not in stopwords_list]
    
    return data_stopped


def create_vector_df(vector_df):
    '''This function converts a dataframe where each instance is the vector for a particular word in the form of a
    list into a dataframe where each column represents a dimension from the vector. The function assumes all vectors
    are the same shape. Designed to be used as a helper function with the get_spacy_vectors function.
    
    vector_df: a column from a pandas dataframe where each row contains a word vector in the form of a python list
    
    returns a pandas dataframe where each column represents one of the dimensions from the vector and each row is 
    an individual vector
    
    '''
    
    # List to hold individual, transposed vectors
    df = []
    # Loop through each instance
    for i in list(range(0,len(vector_df.vector))):
        # Put vector into a dataframe and transpose; (n,1) --> (1,n)
        vectors = pd.DataFrame(vector_df.vector[i]).T
        # Store individual, transposed vector in a list for subsequent concatenation
        df.append(vectors)
    # Combine all individual, transposed vectors into a single dataframe
    vecs_df = pd.concat(df, axis=0)
    
    return vecs_df


def get_spacy_vectors(text):
    '''Function designed to grab the english SpaCy vectors for the words in a specific corpus and return them in a
    dataframe where each row corresponds to a specific vector and where each column represents one of the dimensions 
    of the vector.
    
    text: tokenized string; designed to take in a column from a pandas dataframe with tokenized texts as intances
    
    returns a pandas dataframe where each column represents one of the dimensions from the vector and each row is 
    an individual vector
    
    '''
    # Instantiate a dictionary to hold word vectors
    word_vecs = {}
    
    # Loop through each tokenized document in the input text
    for doc in text:
        # Loop through each token in the text instance
        for word in doc:
            # Conditional to detemine if the text instance word is in SpaCy's embedded vocabulary
            if word in nlp.vocab:
                # Create a word:vector entry in the word_vecs dictionary
                word_vecs[word] = nlp.vocab[word].vector
            # Discard the text instance word if not in SpaCy's embedded vocabulary
            else:
                continue
    # Create a dataframe where each instance holds the word's vector as a list
    vec_df = pd.DataFrame()
    vec_df['vector'] = [word_vecs[x] for x in list(word_vecs.keys())]
    vec_df['word'] = [x for x in list(word_vecs.keys())]
    
    # Use the create_vector_df helper function to convert the vectors stored as lists into their own dataframe
    vectors_df = create_vector_df(vec_df)
    
    return vectors_df


def pca_3d(vecs1, vecs2, vecs3, title, elev=None, rotate=None):
    ''' Function designed to reduce (via PCA) the high-dimensional word vectors to three dimensions with the intention
    of plotting their relationship. 
    
    vecs1, vecs2, vecs3: pandas dataframe containing the word vectors where each row represents and entire vector
    and each column represents a dimension of the vectors
    
    title: string denoting the title of the plot
    
    elev: integer pertaining to what elevation to apply to the view of the 3d plot
    
    rotate: integer pertaining to what angle to apply to the view of the 3d plot
    
    returns a three dimensional plot at the specified viewing angle where vecs1 is red, vecs2 is green, and vecs3 is 
    blue
    
    '''
    
    # Perform PCA on the three vector dfs
    pca = PCA(n_components=3)
    vecs1_pca = pca.fit_transform(vecs1)
    vecs2_pca = pca.fit_transform(vecs2)
    vecs3_pca = pca.fit_transform(vecs3)
    
    # Create a new dataset for the 3 principal components for the three PCA'd vectors
    vecs1_results = pd.DataFrame(data = vecs1_pca , columns = ['PC1', 'PC2', 'PC3'])
    vecs2_results = pd.DataFrame(data=vecs2_pca, columns=['PC1', 'PC2', 'PC3'])
    vecs3_results = pd.DataFrame(data=vecs3_pca, columns=['PC1', 'PC2', 'PC3'])
    
    # Plot the figure at the specified viewing angle
    fig = plt.figure(figsize=(15,10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(vecs1_results['PC1'], vecs1_results['PC2'], vecs1_results['PC3'], c='crimson', label='Conservative')
    ax.scatter(vecs2_results['PC1'], vecs2_results['PC2'], vecs2_results['PC3'], c='green', label='Moderate')
    ax.scatter(vecs3_results['PC1'], vecs3_results['PC2'], vecs3_results['PC3'], c='darkblue', label='Liberal')
    ax.set_title(title)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.view_init(elev, rotate)
    plt.legend()
    plt.show()
 

    

def preprocess (data_df, fraction, stop=False, max_seq_len=1000):
    '''Process data for kera text classification. Takes in a dataframe containing raw text and their corresponding
    labels. Returns a tokenized and padded sequences for training and test sets, along with their one-hot encoded
    labels.
    
    data_df: pandas dataframe containing the raw texts and labels
    
    fraction: float between 0 and 1 pertaining to the percentage of the data_df to sample and preprocess
    
    stop: boolean to include stopwords in filter; default == False
    
    max_seq_len: integer specifying the maximum length of the tokenized sequences during padding; default == 1000
    
    returns tokenized and padded sequences for training and test sets, along with one-hot encoded labels in the order
    of train_x, test_x, train_y, test_y
    
    '''
    # Sample the data according to specified percentage
    data_sample = data_df.sample(frac=fraction, random_state=16)
    
    # Split the data into a training and test set
    train_X, test_X, train_Y, test_Y = train_test_split(data_sample.text, data_sample.ideology_class, test_size=0.2,
                                                       random_state=16)
    
    ## Convert labels to dummie values (formatting for keras model use of categorical_crossentropy)
    train_y = pd.get_dummies(train_Y)
    test_y = pd.get_dummies(test_Y)
    
    # Conditional to deal with if the data should have stopwords removed
    if stop == True:
        # Get a list of training instances
        list_train_texts = train_X.values
        # Instantiate a tokenizer to convert words to tokens, filter out stopwords, and lowercase all tokens
        tokenizer = text.Tokenizer(num_words=26000, lower=True, filters=joined_stop)
        # Fit the tokenizer to the training data
        tokenizer.fit_on_texts(list(list_train_texts))
        # Convert the tokenized words to a sequence (for recurrent/bidirectional modeling)
        list_tokenized_train = tokenizer.texts_to_sequences(list_train_texts)
        # Get a list of testing instances
        list_test_texts = test_X.values
        # Fit the tokenizer on the testing data
        tokenizer.fit_on_texts(list(list_test_texts))
        # Convert the tokenized words to a sequence (for recurrent/bidirectional modeling)
        list_tokenized_test = tokenizer.texts_to_sequences(list_test_texts)
        
        # Pad the tokenized sequences to the specified length
        train_x_tokenized = sequence.pad_sequences(list_tokenized_train, maxlen=max_seq_len)
        test_x_tokenized = sequence.pad_sequences(list_tokenized_test, maxlen=max_seq_len)
        
    # Conditional to deal with if the data should not have stopwords removed
    else:
        # Get a list of training instances
        list_train_texts = train_X.values
        # Instantiate a tokenizer to convert words to tokens and lowercase all tokens
        tokenizer = text.Tokenizer(num_words=26000, lower=True)
        # Fit the tokenizer to the training data
        tokenizer.fit_on_texts(list(list_train_texts))
        # Convert the tokenized words to a sequence (for recurrent/bidirectional modeling)
        list_tokenized_train = tokenizer.texts_to_sequences(list_train_texts)
        # Get a list of testing instances
        list_test_texts = test_X.values
        # Fit the tokenizer on the testing data
        tokenizer.fit_on_texts(list(list_test_texts))
        # Convert the tokenized words to a sequence (for recurrent/bidirectional modeling)
        list_tokenized_test = tokenizer.texts_to_sequences(list_test_texts)
        
        # Pad the tokenized sequences to the specified length
        train_x_tokenized = sequence.pad_sequences(list_tokenized_train, maxlen=max_seq_len)
        test_x_tokenized = sequence.pad_sequences(list_tokenized_test, maxlen=max_seq_len)
    
    return train_x_tokenized, test_x_tokenized, train_y, test_y



def model_performance(model_history):
    '''Create two subplots to display the loss and accuracy changes during model training for both the training and
    validation sets.
    
    model_history: keras model.fit object storing the model performance metrics across training epochs
    
    returns two subplots displaying the loss and accuracy of the model across training epochs for both the training
    and validation sets. Training values are displayed in red and validation values are displayed in blue.
    '''
    # Create a variable to work with the model history
    history_dict = model_history.history
    # Isolate the model's loss measures over the course of training
    loss_values = history_dict['loss']
    # Isolate the model's loss measures over the course of validation
    val_loss_values = history_dict['val_loss']
    
    # Variable to set the plot's X values
    epochs = range(1, len(loss_values) + 1)
    
    # Instantiate the subplots
    fig, ax = plt.subplots(1, 2, figsize=[25,10])
    
    # Format the training/validation loss plot
    ax[0].plot(epochs, loss_values, 'red', label='Training loss')
    ax[0].plot(epochs, val_loss_values, 'blue', label='Validation loss')
    ax[0].set_title('Training and Validation loss', fontsize=20)
    ax[0].set_xlabel('Epochs', fontsize=16)
    ax[0].set_ylabel('Loss', fontsize=16)
    ax[0].legend()
    
    
    # Isolate the model's accuracy measures over the course of training
    acc_values = history_dict['acc']
    # Isolate the model's accuracy measures over the course of validation
    val_acc_values = history_dict['val_acc'] 
    
    # Format the training/validation accuracy plot
    ax[1].plot(epochs, acc_values, 'red', label='Training acc')
    ax[1].plot(epochs, val_acc_values, 'blue', label='Validation acc')
    ax[1].set_title('Training and Validation accuracy', fontsize=20)
    ax[1].set_xlabel('Epochs', fontsize=16)
    ax[1].set_ylabel('Accuracy', fontsize=16)

    plt.legend()

    plt.show()
    
    
    
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Purples):
    
    '''Create a confusion matrix to visualize model performance on testing data.
    
    cm: an SKLearn confusion matrix
    classes: list of class names to use as x and y tick labels; list of strings
    normalize: boolean to normalize values (integers for count --> percentage)
    cmap: matplotlib color map to use for plot
    
    returns a matplotlib plot of a confusion matrix
    
    '''
    #Add Normalization Option
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(cm)

    # Create/format confusion matrix plot
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
