# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import TinyCifarCNN

def main():
    device = (
        torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cuda") if torch.cuda.is_available()
        else torch.device("cpu")
    )
    print(f"Using device: {device}")

    tfm_train = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010)),
    ])
    tfm_test = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010)),
    ])

    train_ds = datasets.CIFAR10("./data", train=True, download=True, transform=tfm_train)
    test_ds  = datasets.CIFAR10("./data", train=False, download=True, transform=tfm_test)

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0)
    test_loader  = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)

    model = TinyCifarCNN(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    def evaluate():
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(1)
                correct += (pred == y).sum().item()
                total += y.numel()
        return correct/total

    EPOCHS = 3  
    for epoch in range(1, EPOCHS+1):
        model.train()
        total_loss = 0
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            loss = criterion(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            if (i + 1) % 50 == 0:  # 每50个batch打印一次
                print(f"Epoch {epoch:02d} [{i+1:4d}/{len(train_loader):4d}] loss={total_loss/(i+1):.4f}")
        
        acc = evaluate()
        print(f"Epoch {epoch:02d} complete, test acc={acc:.4f}")

    torch.save({
        "state_dict": model.state_dict(),
        "classes": train_ds.classes
    }, "cnn_cifar10.pt")
    print("Model saved to cnn_cifar10.pt")

if __name__ == "__main__":
    main()
