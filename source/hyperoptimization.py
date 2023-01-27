from call_hyperopt import get_best_hyper
import torch
from torch import nn
from rnn_model import LSTMModel, VocabSizes
from dataloader import get_loaders
from torchtext.data.utils import get_tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# load data
tokenizer = get_tokenizer('basic_english')
vocab_sizes = VocabSizes(tokenizer)
vocab_size, vocab_text = vocab_sizes.get_vocab_size_text()
label_dict = vocab_sizes.get_label_dict()
amount_of_categories = len(label_dict)
max_length = vocab_sizes.get_max_len()

train_loader, val_loader, test_loader, class_weights_ = get_loaders(batch_size_train=300,
                                                                    batch_size_val=300,
                                                                    batch_size_test=300,
                                                                    test_split=0.1,
                                                                    val_split=0.1,
                                                                    shuffle_dataset=True,
                                                                    random_seed=123)


#%%

def best_hyper(set_of_hyper):
    print(set_of_hyper)
    num_hidden_layers = int(set_of_hyper['num_hidden_layers'])
    size_hidden_layer = int(set_of_hyper['size_hidden_layer'])
    emsize = int(set_of_hyper['size_embed'])
    dropout = float(set_of_hyper['dropout'])
    dropout_lstm = float(set_of_hyper['dropout_lstm'])
    lr = float(set_of_hyper['LR'])
    class_weights = int(set_of_hyper['class_weights'])

    model = LSTMModel(vocab_size=vocab_size,
                      embed_dim=emsize,
                      dropout=dropout,
                      dropout_lstm=dropout_lstm,
                      num_hidden_layers=num_hidden_layers,
                      size_hidden_layer=size_hidden_layer,
                      classes=amount_of_categories).to(device)

    if class_weights:
        class_weights = sorted([[label_dict[i[0]], i[1]] for i in class_weights_.items()], key=lambda x: x[0])
        class_weights = torch.Tensor([i[1] for i in class_weights]).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        criterion2 = nn.CrossEntropyLoss(reduction='sum', weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
        criterion2 = nn.CrossEntropyLoss(reduction='sum')

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def train(train_loader, model):
        model.train()
        for batch_idx, (label, text) in enumerate(train_loader):
            if ((batch_idx+1) % 100 == 0) or (len(train_loader) == batch_idx+1):
                print('Finished training {:3d}/{:3d} batches'.format(batch_idx+1, len(train_loader)))

            predicted_label = model(text, text.size(0))
            predicted_label = predicted_label.squeeze(1)
            loss = criterion(predicted_label, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def evaluate(val_loader):
        model.eval()
        loss = 0
        count = 0
        with torch.no_grad():
            for idx, (labels, texts) in enumerate(val_loader):
                predicted_labels = model(texts, texts.size(0))
                loss += criterion2(predicted_labels.squeeze(), labels).item()
                count += len(labels)

        return loss / count
    
    # training session
    epochs = 6
    for epoch in range(epochs):
        print('Initiating epoch: {}/{}'.format(epoch+1, epochs))
        train(iter(train_loader), model)

    # validation session
    val_loss = evaluate(iter(val_loader))
    print("Validation loss:", val_loss)
    
    return val_loss


best_parameter_ = get_best_hyper(best_hyper)
print(best_parameter_)
