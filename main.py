from nltk.stem import PorterStemmer
import string
import math

with open("stopword.txt", "r") as f:
    stop_words = set(line.strip() for line in f)

# create porter stemmer object
porter = PorterStemmer()

terms = []
keys = []
vec_Dic = {}
dicti = {}
dummy_List = []
term_Freq = {}
idf = {}
weight = {}


def filter_r(file_prefix, rows):
    '''function to read and separate the name of the documents and the terms
    present in it to a separate list from the data frame and also create a
    dictionary which has the name of the document as key and the terms present
    in it as the list of strings which is the value of the key'''
    for i in range(rows):
        document_terms = []
        with open(f"{file_prefix}/{i + 1}.txt", "r") as f:
            # read contents of file
            document = f.read()
            # split contents into words
            words = document.split()
            for word in words:
                # convert to lowercase
                word = word.lower()
                # remove punctuation
                word = word.translate(str.maketrans("", "", string.punctuation))
                # remove stopwords
                if word not in stop_words:
                    # apply porter stemming
                    word = porter.stem(word)
                    document_terms.append(word)
                    # add the terms to the list if it is not present else continue
                    if word not in terms:
                        terms.append(word)

        keys.append(f"{i + 1}.txt")
        dicti.update({f"{i + 1}.txt": document_terms})
    print(terms)


def compute_Weight(doc_Count, cols):
    '''Function to compute the weight for each of the terms in the document.
    Here the weight is calculated with the help of term frequency and
    inverse document frequency'''
    for i in terms:
        # initially adding all the elements into the dictionary and initialising
        # the values as zero
        if i not in term_Freq:
            term_Freq.update({i: 0})

    for key, value in dicti.items():
        # to get the number of occurrence of each terms
        for k in value:
            if k in term_Freq:
                term_Freq[k] += 1
                # value incremented by one if the term is found in the documents

    idf = term_Freq.copy()
    # copying the term frequency dictionary to a dictionary named idf which is
    # further neede for computation

    for i in term_Freq:
        term_Freq[i] = term_Freq[i] / cols
        # term frequency is number of occurrence divided by total number of
        # documents

    for i in idf:
        if idf[i] != doc_Count:
            idf[i] = math.log2(cols / idf[i])
            # inverse document frequency log of total number of documents divided
            # by number of occurrence of the terms
        else:
            idf[i] = 0
            # this is to avoid the zero division error

    for i in idf:
        weight.update({i: idf[i] * term_Freq[i]})
        # weight is the product of term frequency and the inverse document
        # frequency

    for i in dicti:
        for j in dicti[i]:
            dummy_List.append(weight[j])

        copy = dummy_List.copy()
        vec_Dic.update({i: copy})
        dummy_List.clear()
        # above operations performed


def get_Weight_For_Query(query):
    '''function to get the weight for each terms present in the query, here we
    consider the term frequency as the weight of the terms'''

    ps = PorterStemmer()

    # apply stemming to each term in the query
    query = [ps.stem(term) for term in query]

    query = [word for word in query if word not in stop_words]
    query_Freq = {}
    # initialisation of the dictionary with query terms as key and its weight as
    # the values

    for i in terms:
        # initially adding all the elements into the dictionary and initialising
        # the values as zero
        if i not in query_Freq:
            query_Freq.update({i: 0})

    for val in query:
        # to get the number of occurrence of each terms
        if val in query_Freq:
            query_Freq[val] += 1
            # value incremented by one if the term is found in the documents

    for i in query_Freq:
        query_Freq[i] = query_Freq[i] / len(query)
        # term frequency obtained by dividing the number of occurrence of terms by
        # total number of terms in the query

    return query_Freq
    # return the dictionary in which the key is the term and the value is the
    # weight


def similarity_Computation(query_Weight):
    ''' Function to calculate the similarity measure in which the weight of the
    query and the document is multiplied in the numerator and the weight is
    squared and squareroot is taken the weights of the query and document'''

    numerator = 0
    denomi1 = 0
    denomi2 = 0
    # initialisation of the variables with zero which is needed for computation

    similarity = {}
    # initialisation of dictionary which has the name of document as key and the
    # similarity measure as value

    for document in dicti:
        for terms in dicti[document]:
            # cosine similarity is calculated

            numerator += weight[terms] * query_Weight[terms]
            denomi1 += weight[terms] * weight[terms]
            denomi2 += query_Weight[terms] * query_Weight[terms]
            # the summation values of the weight is calculated and later they are
            # divided

        if denomi1 != 0 and denomi2 != 0:
            # to avoid the zero division error

            simi = numerator / (math.sqrt(denomi1) * math.sqrt(denomi2))
            similarity.update({document: simi})
            # dictionary is updated

            numerator = 0
            denomi2 = 0
            denomi1 = 0
            # reinitialisation of the variables to zero

    return (similarity)
    # the dictionary containing similarity measure as the value


def prediction(similarity, doc_count):
    '''Function to predict the document which is relevant to the query '''
    ans = 0
    if similarity != 0:
        ans = max(similarity, key=similarity.get)
        print(ans, "is the most relevant document")
    # to print the name of the document which is most relevant

    print("ranking of the documents")
    alpha = 0.05
    for i in range(doc_count):
        if similarity != 0:
            ans = max(filter(lambda x: similarity[x] >= alpha, similarity), key=lambda x: similarity[x])
            print(ans, "rank is", i + 1)
        # to print the document name and its rank

        if ans != 0:
            similarity.pop(ans)



def main():
    rows = 30
    # to get the number of rows

    filter_r("Dataset", rows)
    # function call to read and separate the name of the documents and the terms
    # present in it to a separate list  from the data frame and also create a
    # dictionary which has the name of the document as key and the terms present
    # in it as the list of strings  which is the value of the key

    cols = len(terms)

    compute_Weight(rows, cols)
    # Function to compute the weight for each of the terms in the document.
    # Here the weight is calculated with the help of term frequency and inverse
    # document frequency

    print("Enter the query")
    query = input()

    query = query.split(' ')
    # splitting the query as a list of strings

    query_Weight = get_Weight_For_Query(query)
    # function call to get the weight for each terms present in the query, here we
    # consider the term frequency as the weight of the terms'''

    similarity = similarity_Computation(query_Weight)
    # Function call to calculate the similarity measure in which the weight of the
    # query and the document are multiplied in the numerator and the weight is
    # squared and squareroot is taken the weights of the query and document

    prediction(similarity, rows)
    # Function call to predict the document which is relevant to the query


main()
