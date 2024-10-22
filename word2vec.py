import pandas as pd
import nltk

def create_files(input_file = "./input/raw_input.csv", title_output = "id_title.csv", overview_output = "id_overview.csv", genre_output = "id_genre.csv"):
    data = pd.read_csv(input_file)

    data[['id', 'title']].to_csv(title_output, index=False)
    data[['id', 'overview']].to_csv(overview_output, index=False)
    data[['id', 'genres']].to_csv(genre_output, index=False)

def lemmatize_words(data):
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    data['overview'] = data['overview'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()]))
    return data

def stem_words(data):
    from nltk.stem import PorterStemmer
    stemmer = PorterStemmer()
    data['overview'] = data['overview'].apply(lambda x: ' '.join([stemmer.stem(word) for word in x.split()]))
    return data

def preprocess(lemmatize = False, stem = False, input_file = "id_overview.csv", output_file = "overview_preprocessed.csv"):
    data = pd.read_csv(input_file)

    overview = data['overview']

    # remove all non-alphanumeric characters from the overview column
    overview = overview.str.replace(r'\W', ' ')

    # remove any special characters from the overview column
    overview = overview.apply(lambda x: ''.join(e if e.isalnum() else ' ' for e in x))

    # lowercase the overview column
    overview = overview.str.lower()

    # remove the stopwords from the overview column
    stop_words = set(nltk.corpus.stopwords.words('english'))
    overview = overview.apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in stop_words]))

    # lemmatize the overview column
    if lemmatize:
        overview = lemmatize_words(overview)

    # stem the overview column
    if stem:
        overview = stem_words(overview)

    data['overview'] = overview

    data.to_csv(output_file, index=False)

def genre_OHE(input_file = "id_genre.csv", output_file = "id_genre_OHE.csv"):
    data = pd.read_csv(input_file)

    one_hot_unique = data['genres'].str.get_dummies(sep=',').columns.str.strip().unique()
    for i in one_hot_unique:
        data[i] = data['genres'].str.contains(i).astype(int)
    data.drop('genres', axis=1, inplace=True)

    data.to_csv(output_file, index=False)

def create_dict(input_file = "overview_preprocessed.csv", output_file = "dict.csv"):
    data = pd.read_csv(input_file)

    overview = data['overview']

    dict = {}
    counter = 0
    for i in overview:
        for j in i.split():
            if j not in dict:
                dict[j] = counter
                counter += 1

    pd.DataFrame(list(dict.items()), columns=['word', 'id']).to_csv(output_file, index=False)

def fill_overview_dict(binary = True, dict_input_file = "dict.csv", overview_input_file = "overview_preprocessed.csv", output_file = "id_filled_dict.csv"):
    dict = pd.read_csv(dict_input_file)

    words = dict['word']
    words_df = pd.DataFrame(columns=words)

    overview = pd.read_csv(overview_input_file)
    ids = overview['id']

    if binary:
        for i in range(len(overview)):
            row = overview.iloc[i]
            row_dict = {}
            for j in row['overview'].split():
                row_dict[j] = 1
            words_df = words_df._append(row_dict, ignore_index=True)
    else:
        for i in range(len(overview)):
            row = overview.iloc[i]
            row_dict = {}
            for j in row['overview'].split():
                if j not in row_dict:
                    row_dict[j] = 1
                else:
                    row_dict[j] += 1
            words_df = words_df._append(row_dict, ignore_index=True)

    # replace NaN values with 0
    words_df.fillna(0, inplace=True)
    
    # reorder the columns
    words_df['id'] = ids
    cols = words_df.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    words_df = words_df[cols]

    words_df.to_csv(output_file, index=False)

def create_word2vec(title_input_file = "id_title.csv", genre_input_file = "id_genre_OHE.csv", dict_input_file = "id_filled_dict.csv", output_file = "./output/word2vec.csv"):
    titles = pd.read_csv(title_input_file)
    genres = pd.read_csv(genre_input_file)
    filled_dict = pd.read_csv(dict_input_file)

    overview_genre = pd.merge(filled_dict, genres, on='id')
    overview_genre_title = pd.merge(overview_genre, titles, on='id')

    overview_genre_title.to_csv(output_file, index=False)

if __name__ == "__main__":
    create_files()
    preprocess()
    genre_OHE()
    create_dict()
    fill_overview_dict(binary=False)
    create_word2vec()

#####################################################################################################
# THE RESULTING FILE SHOULD LOOK LIKE THIS:                                                         #
#####################################################################################################
# id        , word1 , word2 , word3 , ...   , genre1    , genre2    , ..., title                    #
#####################################################################################################
# movie id  , binary numbers/counts , ...   , OHE values            , ..., "The Movie Title"        #
# movie id  , binary numbers/counts , ...   , OHE values            , ..., "The Movie Title"        #
# ...                                                                                               #
#####################################################################################################
#    i.e. the first column is the movie id, the next columns are the binary numbers/counts of the   #
# words in the overview, the next columns are the one-hot encoded values of the genres, and the     #
# last column is the title of the movie.                                                            #
#    For further process, do not include the index column. It is stored to avoid confusion if       #
# the original data is needed.                                                                      #
#    Title column is the target column.                                                             #
#    Rest of the columns are the features.                                                          #
#####################################################################################################