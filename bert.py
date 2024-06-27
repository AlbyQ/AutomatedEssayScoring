import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertModel
from transformers import BertTokenizer

train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')

# train_data = train_data.head(100)
# test_data = test_data.head(100)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def tokenize_data(data, max_length=512):
    return tokenizer(
        data['full_text'].tolist(),
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )


train_tokens = tokenize_data(train_data)
test_tokens = tokenize_data(test_data)

y_train = train_data['score'].values

train_idx, val_idx = train_test_split(range(len(y_train)), test_size=0.2, random_state=42)
train_tokens, val_tokens = {key: val[train_idx] for key, val in train_tokens.items()}, {key: val[val_idx] for key, val
                                                                                        in train_tokens.items()}
y_tr, y_val = y_train[train_idx], y_train[val_idx]


class BertRegressor(nn.Module):
    def __init__(self, pretrained_model_name='bert-base-uncased'):
        super(BertRegressor, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        self.regressor = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooler_output = outputs.pooler_output
        return self.regressor(pooler_output)


model = BertRegressor()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

train_dataset = TensorDataset(train_tokens['input_ids'], train_tokens['attention_mask'],
                              torch.tensor(y_tr, dtype=torch.float))
val_dataset = TensorDataset(val_tokens['input_ids'], val_tokens['attention_mask'],
                            torch.tensor(y_val, dtype=torch.float))
test_dataset = TensorDataset(test_tokens['input_ids'], test_tokens['attention_mask'])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)


def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in dataloader:
        input_ids, attention_mask, labels = [b.to(device) for b in batch]
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(outputs.view(-1), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.view(-1), labels)
            total_loss += loss.item()
    return total_loss / len(dataloader)


epochs = 3
best_val_loss = float('inf')

for epoch in range(epochs):
    train_loss = train(model, train_loader, optimizer, criterion, device)
    val_loss = evaluate(model, val_loader, criterion, device)
    print(f"Epoch {epoch + 1}/{epochs}")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Validation Loss: {val_loss:.4f}")
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'result/best_model.pt')

model.load_state_dict(torch.load('result/best_model.pt'))


def predict(model, dataloader, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask = [b.to(device) for b in batch]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions.extend(outputs.view(-1).tolist())
    return predictions


y_test_pred = predict(model, test_loader, device)
y_test_pred_rounded = np.round(y_test_pred).astype(int)

submission = pd.DataFrame({
    'essay_id': test_data['essay_id'],
    'score': y_test_pred_rounded
})
submission.to_csv('result/submission.csv', index=False)
