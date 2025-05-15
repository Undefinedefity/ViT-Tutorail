import hydra
from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from transformers import ViTForImageClassification, ViTConfig, AdamW

@hydra.main(config_path=".", config_name="config")
def main(cfg: DictConfig):
    # 固定随机种子
    torch.manual_seed(cfg.seed)

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)),
    ])
    train_dataset = CIFAR10(root=cfg.data.root, train=True,
                            transform=transform, download=cfg.data.download)
    train_loader = DataLoader(train_dataset, batch_size=cfg.data.batch_size,
                              shuffle=True, num_workers=cfg.data.num_workers)

    # 加载 ViT 模型
    vit_config = ViTConfig.from_pretrained(cfg.model.name,
                                           num_labels=cfg.model.num_labels,
                                           hidden_dropout_prob=cfg.model.dropout)
    model = ViTForImageClassification.from_pretrained(cfg.model.name,
                                                      config=vit_config)
    model.to(cfg.train.device)

    # 优化器
    optimizer = AdamW(model.parameters(), lr=cfg.train.lr,
                      weight_decay=cfg.train.weight_decay)

    # 训练循环
    model.train()
    for epoch in range(cfg.train.epochs):
        for step, (imgs, labels) in enumerate(train_loader):
            imgs = imgs.to(cfg.train.device)
            labels = labels.to(cfg.train.device)

            outputs = model(pixel_values=imgs, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % cfg.train.log_steps == 0:
                print(f"[Epoch {epoch+1}/{cfg.train.epochs}] "
                      f"Step {step}/{len(train_loader)} - loss: {loss.item():.4f}")

    # 保存模型
    save_path = hydra.utils.to_absolute_path(f"{cfg.output_dir}/vit_cifar10.pt")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    main()
