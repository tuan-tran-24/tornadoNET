from torch import nn
from torchvision import models

import torch

class cnn_stm(nn.Module):
    """
    Shared ResNet-50 feature extractor with:
      - Sequence head: LSTM + mean logits -> logits
      - Still head:    linear -> logits
    """
    def __init__(
        self,         
        num_classes: int, 
        d_model: int = 512,          # transformer token dimensions
        n_head: int = 8,
        stm_layers: int = 2,
        dropout: float = 0.2,
        max_frames: int = 512,       # max sequence length supported by learned pos embedding
    ):
        super().__init__()

        # Backbone
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        backbone.fc = nn.Identity()  # (N,2048)
        self.backbone = backbone
        
        # Temporal Transformer
        self.frame_projection = nn.Linear(2048, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1,1,d_model))
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_frames + 1, d_model))

        # initializations of cls_token and pos_enc with truncated normal
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.positional_encoding, std=0.02)

        # building block of Transformer encoder
        encoding_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=4*d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )

        # build the STM
        self.temporal_encoder = nn.TransformerEncoder(
            encoding_layer,
            num_layers=stm_layers,
            enable_nested_tensor=False,
            )                                                             

        # fully-connected layers
        self.head_sequence = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )
        
        # fully-connected layers
        self.head_still = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2048, num_classes)
        )

    def _positional(
        self, 
        token_plus_cls: int,
        device: torch.device) -> torch.Tensor:
        """ 
        Return positional embedding [1, token_plus_cls, d_model].
        If the sequence is longer than max_frames, interpolates.
        """
        pe = self.positional_encoding
        if token_plus_cls <= pe.size(1):
            return pe[:,:token_plus_cls,:].to(device)

        # interpolate positional embedding in time (1D)
        # pe: [1, L, D] -> [1, D, L]
        pe_token = pe.transpose(1,2)
        pe_token = F.interpolate(pe_token,
                                 size = token_plus_cls,
                                 mode = "linear",
                                 align_corners=False)
        
        return pe_token.transpose(1,2).to(device)
    
    def forward_sequence(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [T,3,H,W] or [1,T,3,H,W]
        returns: [1,K]
        """
        if x.dim() == 4:          # [T,3,H,W]
            x = x.unsqueeze(0)    # [1,T,3,H,W]

        N, T, C, H, W = x.shape

        # performs convolutional operation on each frame
        features = self.backbone(
            x.reshape(N * T, C, H, W)
        ).reshape(N, T, 2048)  # [N,T,2048]

        # Project to transformer token dim
        token = self.frame_projection(features) #[N,T,d_model]

        # Prepend CLS token
        cls = self.cls_token.expand(N,1,-1) # [N,1,d_model]
        token = torch.cat([cls, token], dim=1) # [N,T+1,d_model]

        # Add positional encoding
        token = token + self._positional(T+1, token.device)  # [N,T+1,d_model]

        # sequential modeling
        out = self.temporal_encoder(token) # [N,T+1,d_model]
        cls_out = out[:,0,:]

        # classification head
        logits_sequence = self.head_sequence(cls_out)
        return logits_sequence # [N,K]

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
        
        # pass stills to backbone and then mean    
        features = self.backbone(x)            # [K,2048]
        logits_still = self.head_still(features)     # [K,C]
        
        # safeguard to ensure same dimension
        if logits_still.shape[0] > 1:
            logits_still = logits_still.mean(dim=0,keepdim=True)
            
        return logits_still           
