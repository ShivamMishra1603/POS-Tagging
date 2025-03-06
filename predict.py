import torch
from token_utils import wordTokenizer, getFeaturesForTarget
import numpy as np

def predict_POS_tags(model, sentence, word_to_index, pos_to_index):

    tokens = wordTokenizer(sentence)
    
    feature_vectors = []
    for idx, token in enumerate(tokens):
        feature_vector = getFeaturesForTarget(tokens, idx, word_to_index)
        feature_vectors.append(feature_vector)
    
    feature_tensor = torch.tensor(np.array(feature_vectors), dtype=torch.float32)
    
    model.eval()
    with torch.no_grad():
        outputs = model(feature_tensor)
    
    _, predicted_classes = torch.max(outputs, dim=1)
    
    predicted_pos_tags = [list(pos_to_index.keys())[predicted_classes[i]] for i in range(len(predicted_classes))]
    
    return list(zip(tokens, predicted_pos_tags))
