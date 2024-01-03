import torchtext


def process_dataset(dataset):
    pairs = []
    for _, question, answer, _ in dataset:
        answer = answer[0]
        pair = question.lower(), answer.lower()
        pairs.append(pair)
    return pairs


def load_datasets(path):
    train_dataset, valid_dataset = torchtext.datasets.SQuAD1(path)
    train_dataset = process_dataset(train_dataset)
    valid_dataset = process_dataset(valid_dataset)
    return train_dataset, valid_dataset


def preview_dataset(dataset, n = 5):
    for i in range(n):
        question, answer = dataset[i]
        print(f'Question: "{question}"')
        print(f'Answer: "{answer}"\n')


train_dataset, valid_dataset = load_datasets('data')

print(f'len(train_dataset)={len(train_dataset)}, len(valid_dataset)={len(valid_dataset)}')
train_dataset = train_dataset[:8000]
valid_dataset = valid_dataset[:2000]
print(f'len(train_dataset)={len(train_dataset)}, len(valid_dataset)={len(valid_dataset)}')
