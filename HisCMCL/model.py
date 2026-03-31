import timm
import torch
from torch import nn, einsum
from einops import rearrange
from torchvision.models import DenseNet121_Weights, ResNet18_Weights
import torchvision.models as models
import torch.nn.functional as F


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads  # 64*8 = 512
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = self.attend(dots)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class attn_block(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.attn = PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))
        self.ff = PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))

    def forward(self, x):
        x = self.attn(x) + x
        x = self.ff(x) + x
        return x


class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.densenet121(weights=DenseNet121_Weights.DEFAULT)
        self.model = nn.Sequential(*list(self.model.children())[:-1])

        for p in self.model.parameters():
            p.requires_grad = True

    def forward(self, x):
        x = self.model(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        return x


class ImageEncoder_Resnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet50(pretrained=True)
        self.model = nn.Sequential(*list(self.model.children())[:-1])

        for p in self.model.parameters():
            p.requires_grad = True

    def forward(self, x):
        x = self.model(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        return x


class ImageEncoder_VIT(nn.Module):
    def __init__(
            self, model_name="vit_base_patch32_224", pretrained=True, trainable=True
    ):
        super().__init__()
        self.model = timm.create_model(
            model_name, pretrained, num_classes=0, global_pool="avg"
        )
        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        return self.model(x)


class ImageEncdoer_res18(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.model = nn.Sequential(*list(self.model.children())[:-1])

        for p in self.model.parameters():
            p.requires_grad = True

    def forward(self, x):
        x = self.model(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        return x


class ImageEncdoer_res101(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet101(pretrained=True)
        self.model = nn.Sequential(*list(self.model.children())[:-1])

        for p in self.model.parameters():
            p.requires_grad = True

    def forward(self, x):
        x = self.model(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        return x


class ProjectionHead(nn.Module):
    def __init__(self, embedding_dim, projection_dim, dropout=0.):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)

        return x


class MultiScaleImageEncoder(nn.Module):
    def __init__(self, encoder_name="densenet121", scales=[224, 336, 448], use_attention=True, memory_efficient=False):
        super().__init__()
        # 记录参数
        self.scales = scales
        self.use_attention = use_attention
        self.num_scales = len(scales)
        self.memory_efficient = memory_efficient

        if encoder_name == "densenet121":
            self.base_encoder = models.densenet121(weights=DenseNet121_Weights.DEFAULT)
            self.base_encoder = nn.Sequential(*list(self.base_encoder.children())[:-1])
            self.feature_dim = 1024
        elif encoder_name == "resnet50":
            self.base_encoder = models.resnet50(pretrained=True)
            self.base_encoder = nn.Sequential(*list(self.base_encoder.children())[:-1])
            self.feature_dim = 2048
        elif encoder_name == "vit":
            self.base_encoder = timm.create_model(
                "vit_base_patch32_224", pretrained=True, num_classes=0, global_pool="avg"
            )
            self.feature_dim = 768
        elif encoder_name == "res18":
            self.base_encoder = models.resnet18(weights=ResNet18_Weights.DEFAULT)
            self.base_encoder = nn.Sequential(*list(self.base_encoder.children())[:-1])
            self.feature_dim = 512
        elif encoder_name == "res101":
            self.base_encoder = models.resnet101(pretrained=True)
            self.base_encoder = nn.Sequential(*list(self.base_encoder.children())[:-1])
            self.feature_dim = 2048

        self.fusion = nn.Sequential(
            nn.Linear(self.feature_dim * self.num_scales, self.feature_dim),
            nn.LayerNorm(self.feature_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        if self.use_attention:
            self.scale_attention = nn.Sequential(
                nn.Linear(self.feature_dim * self.num_scales, self.num_scales),
                nn.Softmax(dim=1)
            )
        
        for p in self.base_encoder.parameters():
            p.requires_grad = True

    def forward(self, x):
        features_list = []

        for i, scale in enumerate(self.scales):
            if scale == 224:
                x_scaled = x
            else:
                x_scaled = F.interpolate(x, size=(scale, scale), mode='bilinear', align_corners=False)

            features = self.base_encoder(x_scaled)
            if not isinstance(features, torch.Tensor):
                features = features.view(features.size(0), -1)
            else:
                features = F.adaptive_avg_pool2d(features, (1, 1))
                features = features.view(features.size(0), -1)
            
            features_list.append(features)

            if self.memory_efficient and i < len(self.scales) - 1:
                del x_scaled
                torch.cuda.empty_cache()

        if self.memory_efficient and len(features_list) > 2:
            feat_list_copy = features_list.copy()
            concat_features = feat_list_copy[0]

            for i in range(1, len(feat_list_copy)):
                concat_features = torch.cat([concat_features, feat_list_copy[i]], dim=1)

                if i < len(feat_list_copy) - 1:
                    torch.cuda.empty_cache()
        else:
            concat_features = torch.cat(features_list, dim=1)

        if self.use_attention:
            attention_weights = self.scale_attention(concat_features)

            if self.memory_efficient:
                weighted_features = features_list[0] * attention_weights[:, 0].unsqueeze(1)
                for i in range(1, len(features_list)):
                    curr_weight = attention_weights[:, i].unsqueeze(1)
                    curr_feature = features_list[i]
                    weighted_feat_i = curr_feature * curr_weight

                    weighted_features = torch.cat([weighted_features, weighted_feat_i], dim=1)

                    del weighted_feat_i, curr_weight
                    if i < len(features_list) - 1:
                        torch.cuda.empty_cache()
            else:
                weighted_features = torch.cat([
                    features_list[i] * attention_weights[:, i].unsqueeze(1) 
                    for i in range(self.num_scales)
                ], dim=1)

            fused_features = self.fusion(weighted_features)

            if self.memory_efficient:
                del weighted_features, attention_weights, concat_features, features_list
                torch.cuda.empty_cache()
        else:
            fused_features = self.fusion(concat_features)

            if self.memory_efficient:
                del concat_features, features_list
                torch.cuda.empty_cache()
        
        return fused_features


class mclSTExp_MLP(nn.Module):
    def __init__(self, temperature, image_embedding, spot_embedding, projection_dim, dropout=0.):
        super().__init__()
        self.x_embed = nn.Embedding(65536, spot_embedding)
        self.y_embed = nn.Embedding(65536, spot_embedding)
        self.image_ecode = ImageEncoder()
        self.image_projection = ProjectionHead(embedding_dim=image_embedding, projection_dim=projection_dim)
        self.spot_projection = ProjectionHead(embedding_dim=spot_embedding, projection_dim=projection_dim)
        self.temperature = temperature

    def forward(self, batch):
        image_features = self.image_ecode(batch["image"])
        spot_features = batch["expression"]
        image_embeddings = self.image_projection(image_features)
        x = batch["position"][:, 0].long()
        y = batch["position"][:, 1].long()
        centers_x = self.x_embed(x)
        centers_y = self.y_embed(y)

        spot_features = spot_features + centers_x + centers_y

        spot_embeddings = self.spot_projection(spot_features)
        cos_smi = (spot_embeddings @ image_embeddings.T) / self.temperature
        label = torch.eye(cos_smi.shape[0], cos_smi.shape[1]).cuda()
        spots_loss = F.cross_entropy(cos_smi, label)
        images_loss = F.cross_entropy(cos_smi.T, label.T)
        loss = (images_loss + spots_loss) / 2.0  # shape: (batch_size)
        return loss.mean()


class mclSTExp_Attention(nn.Module):
    def __init__(self, encoder_name, temperature, image_dim, spot_dim, projection_dim, 
                heads_num, heads_dim, head_layers, dropout=0., 
                use_multiscale=False, scales=[224, 448, 672], scale_attention=True, memory_efficient=False):
        super().__init__()
        self.x_embed = nn.Embedding(65536, spot_dim)
        self.y_embed = nn.Embedding(65536, spot_dim)
        
        # 根据参数选择是否使用多尺度特征提取
        if use_multiscale:
            self.image_encoder = MultiScaleImageEncoder(
                encoder_name=encoder_name,
                scales=scales,
                use_attention=scale_attention,
                memory_efficient=memory_efficient
            )
        else:
            if encoder_name == "resnet50":
                self.image_encoder = ImageEncoder_Resnet()
            elif encoder_name == "densenet121":
                self.image_encoder = ImageEncoder()
            elif encoder_name == "vit":
                self.image_encoder = ImageEncoder_VIT()
            elif encoder_name == "res18":
                self.image_encoder = ImageEncdoer_res18()
            elif encoder_name == "res101":
                self.image_encoder = ImageEncdoer_res101()
        
        self.spot_encoder = nn.Sequential(
            *[attn_block(spot_dim, heads=heads_num, dim_head=heads_dim, mlp_dim=spot_dim, dropout=dropout) for _ in
              range(head_layers)])

        if encoder_name == "densenet121":
            adjusted_image_dim = 1024
        elif encoder_name in ["resnet50", "res101"]:
            adjusted_image_dim = 2048
        elif encoder_name == "vit":
            adjusted_image_dim = 768
        elif encoder_name == "res18":
            adjusted_image_dim = 512
        else:
            adjusted_image_dim = image_dim
            
        self.image_projection = ProjectionHead(embedding_dim=adjusted_image_dim, projection_dim=projection_dim)
        self.spot_projection = ProjectionHead(embedding_dim=spot_dim, projection_dim=projection_dim)

        self.temperature = temperature

    def forward(self, batch):
        image_features = self.image_encoder(batch["image"])
        spot_feature = batch["expression"]
        image_embeddings = self.image_projection(image_features)

        x = batch["position"][:, 0].long()
        y = batch["position"][:, 1].long()
        centers_x = self.x_embed(x)
        centers_y = self.y_embed(y)

        spot_features = spot_feature + centers_x + centers_y
        spot_features = spot_features.unsqueeze(dim=0)

        spot_embeddings = self.spot_encoder(spot_features)
        spot_embeddings = self.spot_projection(spot_embeddings)
        spot_embeddings = spot_embeddings.squeeze(dim=0)

        cos_smi = (spot_embeddings @ image_embeddings.T) / self.temperature
        label = torch.eye(cos_smi.shape[0], cos_smi.shape[1]).cuda()
        spots_loss = F.cross_entropy(cos_smi, label)
        images_loss = F.cross_entropy(cos_smi.T, label.T)
        loss = (images_loss + spots_loss) / 2.0
        return loss.mean()


