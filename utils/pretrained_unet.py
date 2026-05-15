import segmentation_models_pytorch as smp
import torch.nn as nn

def get_model(device):
    print("🧠 กำลังโหลดโมเดล: Standard U-Net (Encoder: efficientnet-b3)")
    
    # 🟢 ใช้ smp.Unet ธรรมดา
    model = smp.Unet(
        encoder_name="efficientnet-b3", # เปลี่ยนเป็น efficientnet-b3, inceptionv4, mit_b2
        encoder_weights="imagenet",     # โหลด Pre-trained Weights
        in_channels=1,                  # Input 1 ช่อง
        classes=1,                      # Output 1 ช่อง (Binary)
        activation=None                 # ปล่อยค่า Logits
    )
    
    # ห่อ Model ให้เข้ากับ Training Loop ของเรา
    class WrappedModel(nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base = base_model
            
        def forward(self, x):
            return {'out': self.base(x)}

    wrapped_model = WrappedModel(model)
    return wrapped_model.to(device)