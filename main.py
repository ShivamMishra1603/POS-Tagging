import os
from token_utils import getConllTags, getLexicalFeatureSet
from model import trainLogReg
from grid_search import gridSearch
from predict import predict_POS_tags
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    current_directory = os.path.dirname(__file__)
    file_path = os.path.join(current_directory, 'data/daily547_3pos.txt')



    wordTagsPerSent = getConllTags(file_path)
    flat_wordTagsPerSent = [item for sublist in wordTagsPerSent for item in sublist]
    token_to_index = {}
    pos_to_index = {}

    for idx, (token, pos) in enumerate(flat_wordTagsPerSent):
        if token not in token_to_index:
            token_to_index[token] = len(token_to_index)
        if pos not in pos_to_index:
            pos_to_index[pos] = len(pos_to_index)

    

    X, y = getLexicalFeatureSet(wordTagsPerSent, token_to_index, pos_to_index)

    X_train, X_dev, y_train, y_dev = train_test_split(X, y, test_size=0.3, random_state=42)

    learning_rate = 0.01
    l2_penalty = 0.01
    model, train_losses, train_accuracies, dev_losses, dev_accuracies = trainLogReg((X_train, y_train), (X_dev, y_dev), learning_rate, l2_penalty)

    learning_rates = [0.1, 1, 10]
    l2_penalties = [1e-5, 1e-3, 1e-1]
    grid_accuracy_table = gridSearch((X_train, y_train), (X_dev, y_dev), learning_rates, l2_penalties)

    sample_sentences = ['The horse raced past the barn fell.', 'For 3 years, we attended S.B.U. in the CS program.', 'Did you hear Sam tell me to "chill out" yesterday? #rude']
    for sentence in sample_sentences:
        predictions = predict_POS_tags(model, sentence, token_to_index, pos_to_index)
        print(f"Predictions for '{sentence}': {predictions}")
