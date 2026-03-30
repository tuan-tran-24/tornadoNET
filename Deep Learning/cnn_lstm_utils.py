from torch import nn
from torchvision import models

import torch

class cnn_lstm(nn.Module):
    """
    Shared ResNet-50 (2048-D frame features) with:
      - Sequence head: LSTM + mean over time -> logits
      - Still head:    linear -> logits
    """
    def __init__(self,
                 num_classes: int,
                 lstm_hidden: int = 128,
                 lstm_layers: int = 1,
                 dropout_p: float = 0.2):
        super().__init__()

        # Backbone
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)   
        backbone.fc = nn.Identity()  # (N,2048)
        self.backbone = backbone
        
        self.lstm = nn.LSTM(
            input_size=2048,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=False,
        )
        
        self.head_sequence = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Linear(lstm_hidden, num_classes),
        )
        
        self.head_still = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Linear(2048, num_classes),
        )

    def forward_sequence(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [T,3,H,W] or [1,T,3,H,W]
        returns: [1,K]
        """
        if x.dim() == 4:          # [T,3,H,W]
            x = x.unsqueeze(0)    # [1,T,3,H,W]

        N, T, C, H, W = x.shape

        # CNN per frame
        features = self.backbone(
            x.reshape(N * T, C, H, W)
        ).reshape(N, T, 2048)  # [N,T,2048]

        out, _ = self.lstm(features)  # [N,T,H]

        # aggregate across time and then average
        h_mean = out.mean(dim=1)   # [N,H]
        logits_sequence = self.head_sequence(h_mean) 
        return logits_sequence        # [N,K]

    def forward_still(self, x: torch.Tensor) -> torch.Tensor:
        """
        x can be:
        [3,H,W] single frame
        [K,3,H,W] bag of frames 
        [1,K,3,H,W] if DataLoader batch_size=1 wraps it

        Return logits [1, num_classes
        """
        if x.dim() == 5: # [1,K,3,H,W]
            x = x.squeeze(0) # [K,3,H,W]
            
        if x.dim() == 3: # [3,H,W]
            x = x.unsqueeze(0) # [1,3,H,W]
            
        features = self.backbone(x)            # [K,2048]
        logits_still = self.head_still(features)     # [K,C]
        
        if logits_still.shape[0] > 1:
            logits_still = logits_still.mean(dim=0,keepdim=True)
            
        return logits_still      