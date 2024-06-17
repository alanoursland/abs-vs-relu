# src/training/train_utils.py

import torch

def train(model, device, train_loader, optimizer, criterion, epoch, scheduler=None, log_interval=100):
    model.train()
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        # print(f"output {output}")
        # print(f"target {target}")
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        if batch_idx % log_interval == 0:
            print(
                f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} "
                f"({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}"
            )
    if scheduler != None:
        scheduler.step()
    train_loss /= len(train_loader.dataset)
    return train_loss


def test_fast(model, X_test, Y_test, criterion, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    sample_count = Y_test.size(0)
    with torch.no_grad():
        output = model(X_test)
        test_loss = criterion(output, Y_test).item() * sample_count
        pred = output.argmax(dim=1, keepdim=True)  # Get the index of the max log-probability
        correct += pred.eq(Y_test.view_as(pred)).sum().item()

    test_loss /= sample_count
    accuracy = 100.0 * correct / sample_count
    print(
        f"{epoch}: Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{sample_count} " f"({accuracy:.2f}%)"
    )
    return test_loss, accuracy

def test(model, test_loader, criterion, epoch, device='cuda'):
    model.eval()
    test_loss = 0
    correct = 0
    sample_count = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item() * data.size(0)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            sample_count += data.size(0)

    test_loss /= sample_count
    accuracy = 100.0 * correct / sample_count
    print(
        f"{epoch}: Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{sample_count} ({accuracy:.2f}%)"
    )
    return test_loss, accuracy
