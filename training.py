import time
from vocab import *


def time_string(dt):
    seconds = int(dt)
    minutes = seconds // 60
    seconds = seconds % 60
    hours = minutes // 60
    minutes = minutes % 60
    if hours > 0:
        return f'{hours:d}:{minutes:02d}:{seconds:02d}'
    return f'{minutes:02d}:{seconds:02d}'
    
    
def dataset_to_tensors(dataset, vocab, device):
    tensors = []
    for question, answer in dataset:
        question_tensor = sentence_to_tensor(vocab, question, device)
        answer_tensor = sentence_to_tensor(vocab, answer, device)
        tensors.append((question_tensor, answer_tensor))
    return tensors        


def create_textfile(filename, firstline):
    with open(filename, 'w') as f:
        f.write(f'{firstline}')


def append_line(filename, line):
    with open(filename, 'a') as f:
        f.write(f'{line}')


def train(vocab, train_dataset, valid_dataset, model, epochs, print_every, learning_rate, batch_size, train_loss_goal, learning_curve_filename):
    
    train_tensors = dataset_to_tensors(train_dataset, vocab, model.device)
    valid_tensors = dataset_to_tensors(valid_dataset, vocab, model.device)
    
    time0 = time.time()
    
    n_train_samples = len(train_tensors)
    n_valid_samples = len(valid_tensors)
    
    model.to(model.device)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    criterion = torch.nn.NLLLoss()

    create_textfile(learning_curve_filename, "Training Loss,Validation Loss\n")

    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}/{epochs}")
        total_training_loss = 0
        total_valid_loss = 0
        loss = 0
    
        model.train()
        for i in range(n_train_samples):
            if i % print_every == 0: 
                dt = time.time() - time0
                print(f'[{time_string(dt)}] Training i={i}/{n_train_samples}')

            src, trg = train_tensors[i]
            output = model(src, trg, src.size(0), trg.size(0))

            current_loss = 0
            for (s, t) in zip(output["decoder_output"], trg): 
                current_loss += criterion(s, t)

            loss += current_loss
            total_training_loss += (current_loss.item() / trg.size(0))

            if i % batch_size == 0 or i == (n_train_samples - 1):
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                loss = 0

        model.eval()
        for i in range(n_valid_samples):
            if i % print_every == 0: 
                dt = time.time() - time0
                print(f'[{time_string(dt)}] Validation i={i}/{n_valid_samples}')
            
            src, trg = valid_tensors[i]
            output = model(src, trg, src.size(0), trg.size(0))

            current_loss = 0
            for (s, t) in zip(output["decoder_output"], trg): 
                current_loss += criterion(s, t)

            total_valid_loss += (current_loss.item() / trg.size(0))

        training_loss = total_training_loss / n_train_samples
        validation_loss = total_valid_loss / n_valid_samples
        dt = time.time() - time0
        print(f"[{time_string(dt)}] Epoch {epoch}/{epochs}, Training Loss = {training_loss:.4f}, Validation Loss = {validation_loss:.4f}")
        append_line(learning_curve_filename, f"{training_loss:.4f},{validation_loss:.4f}\n")

        if training_loss < train_loss_goal:
            print(f'Training was stopped because training_loss < {train_loss_goal:.4f}')
            break