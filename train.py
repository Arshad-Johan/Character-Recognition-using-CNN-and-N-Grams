import random
import time
import numpy as np
import torch
import torchvision
from torchvision import transforms
from model import Model

SAVE_MODEL_PATH = "checkpoint/best_accuracy.pth"

def validate(model, data_loader, device, is_digit=True):
    tp, cnt = 0, 0
    model.eval()
    with torch.no_grad():
        for imgs, labels in data_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            if not is_digit:
                labels = labels - 1 + 10 
            preds = model(imgs)
            preds = torch.argmax(preds, dim=1)
            tp += (preds == labels).sum().item()
            cnt += labels.size(0)
    return tp / cnt if cnt > 0 else 0

def train(opt):
    device = torch.device("cuda:0" if opt.use_gpu and torch.cuda.is_available() else "cpu")
    print("device:", device)

    model = Model(num_classes=36).to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.5,), (0.5,)),
    ])

    trainset_digits = torchvision.datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    trainloader_digits = torch.utils.data.DataLoader(trainset_digits, batch_size=opt.batch_size, shuffle=True)
    
    valset_digits = torchvision.datasets.MNIST(root="./data", train=False, transform=transform, download=True)
    valloader_digits = torch.utils.data.DataLoader(valset_digits, batch_size=opt.batch_size, shuffle=False)

    trainset_letters = torchvision.datasets.EMNIST(root="./data", split='letters', train=True, transform=transform, download=True)
    trainloader_letters = torch.utils.data.DataLoader(trainset_letters, batch_size=opt.batch_size, shuffle=True)
    
    valset_letters = torchvision.datasets.EMNIST(root="./data", split='letters', train=False, transform=transform, download=True)
    valloader_letters = torch.utils.data.DataLoader(valset_letters, batch_size=opt.batch_size, shuffle=False)

    best_eval_acc = 0
    start = time.time()
    print("Number of MNIST training samples:", len(trainset_digits))
    print("Number of EMNIST training samples:", len(trainset_letters))

    for ep in range(opt.num_epoch):
        avg_loss_digits = 0
        avg_loss_letters = 0
        model.train()
        print(f"{ep + 1}/{opt.num_epoch} epoch start")

        # Training mini batch for digits
        for i, (imgs, labels) in enumerate(trainloader_digits):
            imgs, labels = imgs.to(device), labels.to(device)

            preds = model(imgs)
            loss = criterion(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss_digits += loss.item()

            if i > 0 and i % 100 == 0:
                print(f"Digit loss: {avg_loss_digits / 100:.4f}")
                avg_loss_digits = 0

        for i, (imgs, labels) in enumerate(trainloader_letters):
            imgs, labels = imgs.to(device), labels.to(device)
            labels = labels - 1 + 10

            preds = model(imgs)
            loss = criterion(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss_letters += loss.item()

            if i > 0 and i % 100 == 0:
                print(f"Letter loss: {avg_loss_letters / 100:.4f}")
                avg_loss_letters = 0

        if ep % opt.valid_interval == 0:
            acc_digits = validate(model, valloader_digits, device, is_digit=True)
            print(f"Digit eval accuracy: {acc_digits:.4f}")

            acc_letters = validate(model, valloader_letters, device, is_digit=False)
            print(f"Letter eval accuracy: {acc_letters:.4f}")

            combined_acc = (acc_digits + acc_letters) / 2
            if combined_acc > best_eval_acc:
                best_eval_acc = combined_acc
                torch.save(model.state_dict(), SAVE_MODEL_PATH)
                print("Saved best accuracy model")

        print(f"{ep + 1}/{opt.num_epoch} epoch finished. elapsed time: {time.time() - start:.1f} sec")

    print(f"Training finished. Best eval accuracy: {best_eval_acc:.4f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--manual_seed", type=int, default=1111, help="random seed setting")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--num_epoch", type=int, default=20, help="number of epochs to train")
    parser.add_argument("--valid_interval", type=int, default=1, help="validation interval")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--use_gpu", action="store_true", help="use gpu if available")
    opt = parser.parse_args()
    print("args", opt)

    random.seed(opt.manual_seed)
    np.random.seed(opt.manual_seed)
    torch.manual_seed(opt.manual_seed)
    torch.cuda.manual_seed(opt.manual_seed)

    
    train(opt)
