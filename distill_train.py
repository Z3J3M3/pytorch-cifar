'''知识蒸馏训练脚本（ResNet18教师模型→ResNet18学生模型）'''
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from models import __all__ as model_dict

# 蒸馏损失函数（硬标签+软标签）
class DistillationLoss(nn.Module):
    def __init__(self, temperature=3, alpha=0.7):
        super().__init__()
        self.temp = temperature
        self.alpha = alpha
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, student_logits, teacher_logits, labels):
        # 软标签损失（蒸馏损失）
        soft_loss = F.kl_div(
            F.log_softmax(student_logits/self.temp, dim=1),
            F.softmax(teacher_logits/self.temp, dim=1),
            reduction='batchmean'
        ) * (self.temp**2)
        # 硬标签损失（普通分类损失）
        hard_loss = self.cross_entropy(student_logits, labels)
        # 加权融合
        return self.alpha * soft_loss + (1 - self.alpha) * hard_loss

# 加载数据集（与main.py一致）
def load_data(dataset='cifar100', batch_size=128):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)), # CIFAR100均值方差
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    if dataset == 'cifar100':
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
        num_classes = 100
    else:
        raise ValueError("仅支持cifar100数据集")

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
    return trainloader, testloader, num_classes

# 验证函数
def test(model, testloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = 100 * correct / total
    model.train()
    return acc

def main():
    # 参数解析（兼容main.py的参数风格）
    parser = argparse.ArgumentParser(description='知识蒸馏训练CIFAR100')
    parser.add_argument('--dataset', type=str, default='cifar100', help='数据集')
    parser.add_argument('--batch_size', type=int, default=128, help='批次大小')
    parser.add_argument('--epochs', type=int, default=200, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.05, help='学习率')
    parser.add_argument('--gpu', type=int, default=0, help='GPU编号')
    parser.add_argument('--temp', type=float, default=3, help='蒸馏温度')
    parser.add_argument('--alpha', type=float, default=0.7, help='软标签权重')
    parser.add_argument('--teacher_ckpt', type=str, default='checkpoint/resnet18_cifar100.pth', help='教师模型权重路径')
    args = parser.parse_args()

    # 设备配置
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    # 加载数据
    trainloader, testloader, num_classes = load_data(args.dataset, args.batch_size)

    # 初始化模型：教师模型（加载基线ResNet18权重）+ 学生模型（新的ResNet18）
    teacher_model = model_dict['resnet18'](num_classes=num_classes).to(device)
    student_model = model_dict['resnet18'](num_classes=num_classes).to(device)

    # 加载教师模型权重（需先训练基线模型并保存）
    if os.path.exists(args.teacher_ckpt):
        teacher_model.load_state_dict(torch.load(args.teacher_ckpt))
        print(f"成功加载教师模型：{args.teacher_ckpt}")
    else:
        raise FileNotFoundError(f"教师模型权重不存在！请先训练基线resnet18并保存到{args.teacher_ckpt}")
    
    # 教师模型设为评估模式（不训练）
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False

    # 损失函数、优化器
    criterion = DistillationLoss(temperature=args.temp, alpha=args.alpha)
    optimizer = optim.SGD(student_model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 训练循环
    print("开始知识蒸馏训练...")
    best_acc = 0.0
    for epoch in range(args.epochs):
        student_model.train()
        running_loss = 0.0
        for i, (images, labels) in enumerate(trainloader):
            images, labels = images.to(device), labels.to(device)
            
            # 教师模型输出（无梯度）
            with torch.no_grad():
                teacher_logits = teacher_model(images)
            # 学生模型输出
            student_logits = student_model(images)
            
            # 计算损失
            loss = criterion(student_logits, teacher_logits, labels)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # 验证精度
        test_acc = test(student_model, testloader, device)
        # 学习率调度
        scheduler.step()
        
        # 保存最优模型
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(student_model.state_dict(), f'checkpoint/resnet18_distill_cifar100.pth')
        
        # 打印日志
        print(f'Epoch [{epoch+1}/{args.epochs}], Loss: {running_loss/len(trainloader):.4f}, Test Acc: {test_acc:.2f}%, Best Acc: {best_acc:.2f}%')

    print(f"训练完成！最优测试精度：{best_acc:.2f}%，模型保存至 checkpoint/resnet18_distill_cifar100.pth")

if __name__ == '__main__':
    main()
