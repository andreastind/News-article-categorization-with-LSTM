from hyperopt import hp, fmin, tpe, Trials


def get_best_hyper(objective):
    space = {'num_hidden_layers': hp.randint('num_hidden_layers', 1, 8),
             'size_hidden_layer': hp.randint('size_hidden_layer', 20, 100),
             'size_embed': hp.randint('size_embed', 20, 150),
             'dropout': hp.uniform('dropout', 0, 0.5),
             'dropout_lstm': hp.uniform('dropout_lstm', 0, 0.5),
             'LR': hp.choice('LR', [0.01, 0.001]),
             'class_weights': hp.choice('class_weights', [1, 0])}
    trials = Trials()
    best = fmin(objective, space, algo=tpe.suggest, max_evals=30, trials=trials)
    print(trials.trials)
    return best


