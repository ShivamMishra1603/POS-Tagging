import re
import numpy as np

def wordTokenizer(text):
    pattern = r'''
          (?:\#\w+|@\w+)                    # Matches hashtags or @mentions
        | \b(?:[A-Za-z]\.)+[A-Za-z]\b       # Matches capital abbreviations (U.S.A.)
        | \b[a-zA-Z]+(?:'\w+)?\b            # Matches words and contractions (like can't)
        | \d+\.\d+                          # Matches decimal numbers like 5.0
        | \d+                               # Matches whole numbers
        | [.,!?;]                           # Matches common punctuation marks
    '''
    return re.findall(pattern, text, re.VERBOSE)

def getConllTags(filename):
    wordTagsPerSent = [[]]
    sentNum = 0
    with open(filename, encoding='utf8') as f:
        for wordtag in f:
          wordtag=wordtag.strip()
          if wordtag:#still reading current sentence
              (word, tag) = wordtag.split("\t")
              wordTagsPerSent[sentNum].append((word,tag))
          else:#new sentence
              wordTagsPerSent.append([])
              sentNum+=1

    return wordTagsPerSent

def getFeaturesForTarget(tokens, targetI, wordToIndex):

    is_capitalized = 1 if tokens[targetI][0].isupper() else 0
    
    first_letter = tokens[targetI][0]
    if ord(first_letter) > 255:
        first_letter_value = 256
    else:
        first_letter_value = ord(first_letter)    
    first_letter_one_hot = [0]*257
    first_letter_one_hot[first_letter_value] = 1


    word_length = len(tokens[targetI])
    normalized_length = min(word_length, 10) / 10
    

    if targetI > 0:
        prev_word = tokens[targetI - 1]
        prev_word_vector = [0] * len(wordToIndex)
        if prev_word in wordToIndex:
            prev_word_vector[wordToIndex[prev_word]] = 1
    else:
        prev_word_vector = [0] * len(wordToIndex)
    


    current_word = tokens[targetI]
    current_word_vector = [0] * len(wordToIndex)
    if current_word in wordToIndex:
        current_word_vector[wordToIndex[current_word]] = 1
    


    if targetI < len(tokens) - 1:
        next_word = tokens[targetI + 1]
        next_word_vector = [0] * len(wordToIndex)
        if next_word in wordToIndex:
            next_word_vector[wordToIndex[next_word]] = 1
    else:
        next_word_vector = [0] * len(wordToIndex)
    


    featureVector = [is_capitalized] + first_letter_one_hot+[normalized_length] + prev_word_vector + current_word_vector + next_word_vector
    
    return np.array(featureVector)

def getLexicalFeatureSet(tokens_data, wordToIndex, posToIndex):
    X = []
    y = []
    
    for sentence in tokens_data:
        for idx, (token, pos_tag) in enumerate(sentence):
            features = getFeaturesForTarget([token for token, _ in sentence], idx, wordToIndex)
            X.append(features)
            pos_index = posToIndex[pos_tag]
            y.append(pos_index)
    
    return np.array(X), np.array(y)