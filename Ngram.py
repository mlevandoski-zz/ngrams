import collections, math, nltk.tokenize, sys

def find_ngrams(text, gram, ngram_list):
    if gram == 1:
        for i in range(len(text)-1):
            ngram_list.append(text[i])
    elif gram == 2:
        for i in range(len(text)-1):
            ngram_list.append((text[i], text[i+1]))
    elif gram == 3:
        ngram_list.append(('<s>', text[0], text[1]))
        for i in range(len(text)-2):
            ngram_list.append((text[i], text[i+1], text[i+2]))
        ngram_list.append((text[len(text)-2], text[len(text)-1], '</s>'))
    else:
        print "I'm sorry, this program only handles unigrams, bigrams, and trigrams."
    return ngram_list
#end find_ngrams

# returns V, the Vocabulary created from the training set
def preprocessing(train, thres):
    V = [] # V is our vocabulary
    trainlines = []
    trainline = []
    temp = []
    all_words = []
    V.append('<s>')
    V.append('</s>')
    rawlines = train.readlines()
    for line in rawlines:
        trainlines.append('<s>')
        tokens = nltk.wordpunct_tokenize(line)
        words = [w.lower() for w in tokens]
        for word in words:
            all_words.append(word)
            count = collections.Counter(all_words)
            if not word == '.':
                trainlines.append(word)
            if (word not in V and count[word] > thres):
                V.append(word)
        trainlines.append('</s>')
    #end for
    #print trainlines
    return trainlines, V
    #tokenize trainfile

def get_frequencies(dev_unigram, dev_bigram, dev_trigram, uni_freq, bi_freq, tri_freq):
    for token in dev_unigram:
        if token in uni_freq:
            uni_freq[token] += 1
        else:
            uni_freq[token] = 1
    for token in dev_bigram:
        if token in bi_freq:
            bi_freq[token] += 1
        else:
            bi_freq[token] = 1
    for token in dev_trigram:
        if token in tri_freq:
            tri_freq[token] += 1
        else:
            tri_freq[token] = 1
    for key in uni_freq: uni_freq[key] = float(uni_freq[key])/len(uni_freq)
    for key in bi_freq: bi_freq[key] = float(bi_freq[key])/len(bi_freq)
    for key in tri_freq: tri_freq[key] = float(tri_freq[key])/len(tri_freq)
    return uni_freq, bi_freq, tri_freq
#end get_frequencies

# calculates entropy
def calc_reg_entropy(ngram_list, gram):
    entropy = 0
    for g in ngram_list:
        if (gram == 1):
            if g in uni_freq:
                entropy += -(uni_freq[g] * math.log(uni_freq[g], 2))
        elif (gram == 2):
            if g in bi_freq:
                entropy += -(bi_freq[g] * math.log(bi_freq[g], 2))
        elif (gram == 3):
            if g in tri_freq:
                entropy += -(tri_freq[g] * math.log(tri_freq[g], 2))
    return entropy;

def calc_inter_entropy(gram, uni_gram_list, bi_gram_list, tri_gram_list, lambda1, lambda2, lambda3):
    p_x = 0
    ent = 0
    if gram is '2s':
        for i, j in zip(uni_gram_list, bi_gram_list):
            p_x = 0
            if i in uni_freq :
                p_x += lambda1*uni_freq[i]
            if j in bi_freq:
                p_x += lambda2*bi_freq[j]
            if not p_x == 0:
                ent += -(p_x * math.log(p_x, 2))
    else:
        for i, j, k in zip(uni_gram_list, bi_gram_list, tri_gram_list):
            p_x = 0
            if i in uni_freq :
                p_x += lambda1*uni_freq[i]
            if j in bi_freq:
                p_x += lambda2*bi_freq[j]
            if k in tri_freq:
                p_x += lambda3*tri_freq[k]
            if not p_x == 0:
                ent += -(p_x * math.log(p_x, 2))
    return ent;

def get_lambdas(devfile):
    dev_unigram = []
    dev_bigram = []
    dev_trigram = []
    devtext = []
    lambda1 = lambda2 = lambda3 = 0
    low_dev_entropy = curr_dev_entropy = 999
    with open(devfile, "r+") as f:
        for line in f:
            for word in line.split():
                if word not in V:
                    devtext.append('<UNK>')
                else:
                    devtext.append(word)
        dev_unigram = find_ngrams(devtext,1, dev_unigram)
        dev_bigram = find_ngrams(devtext,2, dev_bigram)

        uni_lambda = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
        bi_lambda = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
        tri_lambda = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
        if "2s" in sys.argv:
            for lam1 in uni_lambda:
                for lam2 in bi_lambda:
                    if lam1 + lam2 == 1:
                        curr_dev_entropy = calc_inter_entropy('2s', dev_unigram, dev_bigram, None, lam1, lam2, 0)
                        if curr_dev_entropy > 0 and abs(curr_dev_entropy) < abs(low_dev_entropy):
                            low_dev_entropy = curr_dev_entropy #update
                            lambda1 = lam1
                            lambda2 = lam2
        elif "3s" in sys.argv:
            dev_trigram = find_ngrams(devtext, 3, dev_trigram)
            for lam1 in uni_lambda:
                for lam2 in bi_lambda:
                    for lam3 in tri_lambda:
                        if lam1 + lam2 + lam3 == 1:
                            curr_dev_entropy = calc_inter_entropy('3s', dev_unigram, dev_bigram, dev_trigram, lam1, lam2, lam3)
                            if curr_dev_entropy > 0 and abs(curr_dev_entropy) < abs(low_dev_entropy):
                                low_dev_entropy = curr_dev_entropy #update
                                lambda1 = lam1
                                lambda2 = lam2
                                lambda3 = lam3
    print "lambdas: ", lambda1, lambda2, lambda3
    return lambda1, lambda2, lambda3

#main program
#Ngram.py <1|2|2s|3|3s> <trainfile> <devfile> <testfile>
if '__main__' == __name__:

    # training our n-grams
    train_unigram = []
    train_bigram = []
    train_trigram = []
    uni_freq = {}
    bi_freq = {}
    tri_freq = {}
    traintext = []
    with open(sys.argv[2], "r+") as f:
        text, V = preprocessing(f, 1) # send training file for preprocessing, returns vocabulary
        for word in text:
            if word not in V:
                traintext.append('<UNK>')
            else:
                traintext.append(word);
        train_unigram = find_ngrams(traintext,1, train_unigram)
        train_bigram = find_ngrams(traintext,2, train_bigram)
        train_trigram = find_ngrams(traintext,3, train_trigram)
    uni_freq, bi_freq, tri_freq = get_frequencies(train_unigram, train_bigram, train_trigram, uni_freq, bi_freq, tri_freq)

    # use development file to get lambda for '2s' and '3s'
    if "2s" in sys.argv or "3s" in sys.argv:
        lambda1, lambda2, lambda3 = get_lambdas(sys.argv[3])

    #time for the test file!
    ngram_list = []
    uni_ngram_list = []
    bi_ngram_list = []
    tri_ngram_list = []
    newtext = []
    entropy = 0
    with open(sys.argv[4], "r+") as f:
        #create ngrams
        for line in f: # pass line with proper <UNK> tokens to ngram
            for word in line.split():
                if word not in V:
                    newtext.append('<UNK>')
                else:
                    newtext.append(word)
        if "2s" in sys.argv:
            uni_ngram_list = find_ngrams(newtext,1, uni_ngram_list)
            bi_ngram_list = find_ngrams(newtext,2, bi_ngram_list)
        elif "3s" in sys.argv:
            uni_ngram_list = find_ngrams(newtext,1, uni_ngram_list)
            bi_ngram_list = find_ngrams(newtext,2, bi_ngram_list)
            tri_ngram_list = find_ngrams(newtext, 3, tri_ngram_list)
        else:
            ngram_list = find_ngrams(newtext,int(sys.argv[1]), ngram_list)
        #calculate entropy
        if "2s" in sys.argv:
            entropy = calc_inter_entropy('2s', uni_ngram_list, bi_ngram_list, None, lambda1, lambda2, lambda3)
        elif "3s" in sys.argv:
            entropy = calc_inter_entropy('3s', uni_ngram_list, bi_ngram_list, tri_ngram_list, lambda1, lambda2, lambda3)
        else:
            entropy = calc_reg_entropy(ngram_list, int(sys.argv[1]))
    print "The entropy of the given text is: ", entropy # entropy of the text
    #end main
