import torch
from src.net.net import BadmintonNet

# is count from behind(negative value)
num_cls_task = len(BadmintonNet.end_idx_orders) - 1
end_idx_orders = BadmintonNet.end_idx_orders
idx_LandingX_start = end_idx_orders[-2]

model_acc_names = [
    'HitFrame',
    'Hitter',
    'RoundHead',
    'Backhand',
    'BallHeight',
    'BallType',
    'LandingX',
    'LandingY',
    'HitterLocationX',
    'HitterLocationY',
    'DefenderLocationX',
    'DefenderLocationY',
]

def calculate(preds: torch.Tensor, labels: torch.Tensor, hit_idxs: torch.Tensor, isHits: torch.Tensor):
    with torch.no_grad():
        hit_preds, not_hit_preds, hit_labels, hitFrame_idxs_select = (
            preds[isHits].view(-1, labels.shape[-1]),
            preds[~isHits].view(-1, labels.shape[-1]),
            labels[isHits].view(-1, labels.shape[-1]),
            hit_idxs[isHits].squeeze(dim=0).type(torch.int),
        )   #! label

        start_idx = 0
        cls_acc_tensor = torch.zeros(num_cls_task, dtype=torch.float32, device=preds.device)
        for i, end_idx in enumerate(end_idx_orders[:num_cls_task]):
            cls_acc_tensor[i] = hit_preds[:, start_idx:end_idx].argmax(dim=1)   # classifier
            start_idx = end_idx

        reg_acc_tensor =torch.abs(hit_preds[:, idx_LandingX_start:]).mean(dim=0)  # regration
        for i in range(len(reg_acc_tensor)):
            if i % 2 == 0:
                reg_acc_tensor[i] *= 1280
            else:
                reg_acc_tensor[i] *= 720
        reg_acc_tensor = reg_acc_tensor.int()

        acc_record = torch.hstack(
            [
                cls_acc_tensor, #classifier
                reg_acc_tensor, #regration
            ]
        )

        return acc_record
