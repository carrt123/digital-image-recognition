import torch


def evaluate(test_loader, model, device=None):
    if device is None and isinstance(model, torch.nn.Module):
        # 如果没指定device就使用net的device
        device = list(model.parameters())[0].device
    acc_sum, n = 0.0, 0
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.reshape(-1,1,28,28)
            # 前向传播
            acc_sum += (model(images.to(device)).argmax(dim=1) == labels.to(device)).float().sum().cpu().item()
            model.train()
            n += labels.shape[0]
    return acc_sum / n
