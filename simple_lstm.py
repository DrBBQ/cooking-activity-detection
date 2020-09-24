import torch
from torch import nn, optim

class LSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, n_classes):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes

        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True,
                            bidirectional=False)
        self.dense = nn.Linear(self.hidden_dim, self.n_classes)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, input):
        input = torch.tensor(input)
        lstm_out, _ = self.lstm(input)
        output = self.dense(lstm_out[:,-1])
        return output

    def fit(self, X, y, epochs=10, batch_size=4):
        dataloader = torch.utils.data.DataLoader(list(zip(X, y)),
                                                 batch_size=batch_size,
                                                 shuffle=True,
                                                 drop_last=True)
        # random_sampler = torch.utils.data.RandomSampler()
        # dataloader = torch.utils.data.DataLoader(list(zip(X, y)),
        #                                          batch_sampler=random_sampler)
        optimizer = optim.Adam(self.parameters())
        running_loss = 0
        for epoch in range(epochs):
            for i, data in enumerate(dataloader):
                X_train, y_train = data
                y_pred = self(X_train)
                try:
                    loss = self.loss(y_pred, y_train.long().squeeze())
                except:
                    import ipdb; ipdb.set_trace()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                if i % 1000 == 0:
                    print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0

    def predict(self, X):
        return torch.argmax(self(X), axis=1).detach()
