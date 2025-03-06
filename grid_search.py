from model import trainLogReg

def gridSearch(train_set, dev_set, learning_rates, l2_penalties):
    X_train, y_train = train_set
    X_dev, y_dev = dev_set

    model_accuracies = []

    best_lr = None
    best_l2_penalty = None
    best_accuracy = 0

    for lr in learning_rates:
        for l2 in l2_penalties:
            model, _, _, _, dev_accuracies = trainLogReg((X_train, y_train), (X_dev, y_dev), lr, l2)
            final_dev_accuracy = dev_accuracies[-1]
            model_accuracies.append((lr, l2, final_dev_accuracy))

            if final_dev_accuracy > best_accuracy:
                best_accuracy = final_dev_accuracy
                best_lr = lr
                best_l2_penalty = l2

    accuracy_table = {}
    for lr in learning_rates:
        accuracy_table[lr] = {}
        for l2 in l2_penalties:
            for comb in model_accuracies:
                if comb[0] == lr and comb[1] == l2:
                    accuracy_table[lr][l2] = comb[2]
                    break
    
    print("Dev Set Accuracy for each Hyperparameter Combination:")
    
    header = "          "
    for lr in learning_rates:
        header += f"{lr:10.1f}"
    print(header)
    
    for l2 in l2_penalties:
        row = f"{l2:10.5f}"
        for lr in learning_rates:
            row += f"{accuracy_table[lr][l2]:10.6f}"
        print(row)

    return model_accuracies, best_lr, best_l2_penalty


