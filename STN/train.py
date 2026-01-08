# 训练代码，先运行这个代码，训练完模型再运行vis.py
import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets
from data_utils_model import STNNet, DEVICE, TRANSFORM

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST("./data", train=True, download=True, transform=TRANSFORM),
    batch_size=64, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST("./data", train=False, transform=TRANSFORM),
    batch_size=64, shuffle=False)

model = STNNet().to(DEVICE)
optimizer = optim.SGD(model.parameters(), lr=0.01)
EPOCHS = 20
SAVE_PATH = "./stn_mnist_model.pth"

def train_one_epoch(epoch):
    model.train()
    total_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(DEVICE), target.to(DEVICE)
        
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        if batch_idx % 500 == 0:
            avg_loss = total_loss / (batch_idx + 1)
            print(f'Train Epoch: {epoch} [{batch_idx*len(data)}/{len(train_loader.dataset)} ({100.*batch_idx/len(train_loader):.0f}%)]\tLoss: {avg_loss:.6f}')

def test():
    model.eval()
    test_loss = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.0f}%)\n')

if __name__ == '__main__':
    for epoch in range(1, EPOCHS + 1):
        train_one_epoch(epoch)
        test()
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, SAVE_PATH)
    print(f"Save to: {SAVE_PATH}")