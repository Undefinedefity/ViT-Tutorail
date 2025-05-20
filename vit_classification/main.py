# 导入必要的库
import hydra  # 用于配置管理
from omegaconf import DictConfig  # 用于类型提示
import torch  # PyTorch深度学习框架
from torch.utils.data import DataLoader  # 数据加载器
from torchvision import transforms  # 图像预处理工具
from torchvision.datasets import CIFAR10  # CIFAR10数据集
from transformers import ViTForImageClassification, ViTConfig, AdamW  # HuggingFace的ViT模型和优化器
import os

@hydra.main(config_path=".", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # 固定随机种子，确保实验可重复性
    torch.manual_seed(cfg.seed)

    # 确保数据目录存在
    os.makedirs(cfg.data.root, exist_ok=True)

    # 定义数据预处理流程
    transform = transforms.Compose([
        transforms.Resize(224),  # 将图像调整为224x224大小，ViT模型的标准输入尺寸
        transforms.ToTensor(),   # 将图像转换为PyTorch张量
        transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)),  # 标准化图像数据
    ])
    
    # 加载CIFAR10训练数据集
    train_dataset = CIFAR10(
        root=cfg.data.root,  # 数据集存储路径
        train=True,          # 使用训练集
        transform=transform, # 应用数据预处理
        download=cfg.data.download  # 如果数据集不存在则下载
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg.data.batch_size,  # 每批处理的样本数
        shuffle=True,                    # 随机打乱数据
        num_workers=cfg.data.num_workers # 数据加载的进程数
    )

    # 配置ViT模型参数
    vit_config = ViTConfig.from_pretrained(
        cfg.model.name,  # 预训练模型名称
        num_labels=cfg.model.num_labels,  # 分类标签数（CIFAR10为10）
        hidden_dropout_prob=cfg.model.dropout  # 隐藏层的dropout率
    )
    
    # 加载预训练的ViT模型
    model = ViTForImageClassification.from_pretrained(
        cfg.model.name,
        config=vit_config
    )
    # 将模型移动到指定设备（GPU/CPU）
    model.to(cfg.train.device)

    # 创建优化器
    optimizer = AdamW(
        model.parameters(),  # 模型参数
        lr=cfg.train.lr,    # 学习率
        weight_decay=cfg.train.weight_decay  # 权重衰减
    )

    # 开始训练循环
    model.train()  # 设置模型为训练模式
    for epoch in range(cfg.train.epochs):  # 遍历每个训练轮次
        for step, (imgs, labels) in enumerate(train_loader):  # 遍历每个批次
            # 将数据移动到指定设备
            imgs = imgs.to(cfg.train.device)
            labels = labels.to(cfg.train.device)

            # 前向传播，计算损失
            outputs = model(pixel_values=imgs, labels=labels)
            loss = outputs.loss

            # 反向传播和优化
            optimizer.zero_grad()  # 清空梯度
            loss.backward()        # 计算梯度
            optimizer.step()       # 更新参数

            # 定期打印训练信息
            if step % cfg.train.log_steps == 0:
                print(f"[Epoch {epoch+1}/{cfg.train.epochs}] "
                      f"Step {step}/{len(train_loader)} - loss: {loss.item():.4f}")

    # 保存训练好的模型
    save_path = hydra.utils.to_absolute_path(f"{cfg.output_dir}/vit_cifar10.pt")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    main()  # 运行主函数
