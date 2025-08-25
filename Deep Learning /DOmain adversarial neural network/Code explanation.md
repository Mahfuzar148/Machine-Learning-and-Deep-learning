
---

# 🔧 কোড (পূর্ণ)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm import create_model
import numpy as np
import logging
from typing import Dict, Optional, Tuple, Union

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GradientReversalFunction(torch.autograd.Function):
    """Gradient Reversal Layer for domain adversarial training"""
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


def grad_reverse(x, alpha=1.0):
    """Apply gradient reversal with given alpha"""
    return GradientReversalFunction.apply(x, alpha)


def init_weights(m):
    """Xavier/Glorot initialization for better gradient flow"""
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class MultiSourceDomainDiscriminator(nn.Module):
    """Enhanced domain discriminator for multi-source domain adaptation"""
    
    def __init__(self, in_features: int, hidden_dims: list = None, 
                 num_domains: int = 3, dropout: float = 0.3):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [512, 256]
        
        layers = []
        prev_dim = in_features
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, num_domains))
        
        self.discriminator = nn.Sequential(*layers)
        self.apply(init_weights)
        
    def forward(self, x):
        return self.discriminator(x)


class FeatureBottleneck(nn.Module):
    """Feature bottleneck with strong regularization"""
    
    def __init__(self, in_dim: int, bottleneck_dim: int, dropout: float = 0.5):
        super().__init__()
        
        self.bottleneck = nn.Sequential(
            nn.Linear(in_dim, bottleneck_dim * 2),
            nn.BatchNorm1d(bottleneck_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            nn.Linear(bottleneck_dim * 2, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5)
        )
        self.apply(init_weights)
        
    def forward(self, x):
        return self.bottleneck(x)


class DomainAlignment(nn.Module):
    """Efficient domain alignment module"""
    
    def __init__(self, feature_dim: int, num_domains: int = 3):
        super().__init__()
        self.num_domains = num_domains
        self.feature_dim = feature_dim
        
        # Shared transformation for all domains (more parameter efficient)
        self.domain_transform = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True)
        )
        
        # Domain-specific bias terms (lightweight adaptation)
        self.domain_bias = nn.Parameter(torch.zeros(num_domains, feature_dim))
        
        # Shared feature space projection
        self.shared_projection = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True)
        )
        
        self.apply(init_weights)
        nn.init.normal_(self.domain_bias, std=0.01)
        
    def forward(self, features, domain_labels: Optional[torch.Tensor] = None):
        # Apply shared transformation
        transformed = self.domain_transform(features)
        
        # Apply domain-specific bias if available
        if domain_labels is not None and self.training:
            batch_size = features.size(0)
            device = features.device
            
            # Create domain-specific bias for each sample
            domain_specific_bias = torch.zeros_like(transformed)
            
            for i in range(self.num_domains):
                mask = (domain_labels == i)
                if mask.any():
                    domain_specific_bias[mask] = self.domain_bias[i]
            
            transformed = transformed + domain_specific_bias
        
        return self.shared_projection(transformed)


class asif_clip_dan(nn.Module):
    """ASIF CLIP Domain Adversarial Network for deepfake detection"""
    
    @classmethod
    def from_config(cls, config):
        """Create model instance from benchmark config"""
        backbone_config = config.get('backbone_config', {})
        
        return cls(
            feature_dim=backbone_config.get('feature_dim', 768),
            bottleneck_dim=backbone_config.get('bottleneck_dim', 512),
            domain_hidden=config.get('domain_hidden', [512, 256]),
            num_classes=backbone_config.get('num_classes', 2),
            num_domains=config.get('num_domains', 3),
            pretrained=config.get('pretrained', True),
            num_unfrozen_blocks=config.get('num_unfrozen_blocks', 6),
            mixup_alpha=config.get('mixup_alpha', 0.3),
            entropy_conditioning=config.get('entropy_conditioning', True),
            dropout=backbone_config.get('dropout', 0.3)
        )
    
    def __init__(self, 
                 num_classes: int = 2,
                 feature_dim: int = 768,
                 bottleneck_dim: int = 512,
                 domain_hidden: Union[int, list] = [512, 256],
                 num_domains: int = 3,
                 pretrained: bool = True,
                 num_unfrozen_blocks: int = 6,
                 mixup_alpha: float = 0.3,
                 entropy_conditioning: bool = True,
                 dropout: float = 0.3):
        super().__init__()
        
        # Store parameters
        self.num_classes = num_classes
        self.num_domains = num_domains
        self.mixup_alpha = mixup_alpha
        self.entropy_conditioning = entropy_conditioning
        self.num_updates = 0
        
        # Initialize backbone with proper error handling
        self._init_backbone(pretrained, num_unfrozen_blocks)
        
        # Domain alignment module
        self.domain_alignment = DomainAlignment(self.feature_dim, num_domains)
        
        # Feature bottleneck
        self.bottleneck = FeatureBottleneck(self.feature_dim, bottleneck_dim, dropout)
        
        # Enhanced classifier
        self.classifier = nn.Sequential(
            nn.Linear(bottleneck_dim, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            nn.Linear(bottleneck_dim, bottleneck_dim // 2),
            nn.BatchNorm1d(bottleneck_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            
            nn.Linear(bottleneck_dim // 2, num_classes)
        )
        self.classifier.apply(init_weights)
        
        # Domain discriminator
        if isinstance(domain_hidden, int):
            domain_hidden = [domain_hidden, domain_hidden // 2]
            
        self.domain_discriminator = MultiSourceDomainDiscriminator(
            in_features=bottleneck_dim,
            hidden_dims=domain_hidden,
            num_domains=num_domains,
            dropout=dropout
        )
        
        logger.info(f"Model initialized: {num_domains} domains, "
                   f"{bottleneck_dim}D bottleneck, mixup_alpha={mixup_alpha}")
    
    def _init_backbone(self, pretrained: bool, num_unfrozen_blocks: int):
        """Initialize and configure the backbone network"""
        try:
            # Try CLIP model first
            self.backbone = create_model('vit_base_patch32_224_clip_laion2b', 
                                       pretrained=pretrained)
            logger.info("Loaded CLIP ViT-B/32 backbone")
        except Exception as e:
            logger.warning(f"Could not load CLIP model: {e}")
            try:
                self.backbone = create_model('vit_base_patch32_224', 
                                           pretrained=pretrained)
                logger.info("Loaded standard ViT-B/32 backbone")
            except Exception as e2:
                logger.error(f"Failed to load any backbone: {e2}")
                raise RuntimeError("Could not initialize backbone model")
        
        # Remove head if present
        if hasattr(self.backbone, 'head'):
            self.backbone.head = nn.Identity()
        
        # Get feature dimension
        self._determine_feature_dim()
        
        # Configure trainable layers
        self._configure_trainable_layers(num_unfrozen_blocks)
    
    def _determine_feature_dim(self):
        """Determine the actual feature dimension from backbone"""
        self.backbone.eval()
        with torch.no_grad():
            try:
                dummy_input = torch.randn(1, 3, 224, 224)
                dummy_output = self.backbone(dummy_input)
                if dummy_output.dim() > 2:
                    dummy_output = dummy_output.view(dummy_output.size(0), -1)
                self.feature_dim = dummy_output.size(1)
                logger.info(f"Detected feature dimension: {self.feature_dim}")
            except Exception as e:
                logger.warning(f"Could not determine feature dim: {e}, using default 768")
                self.feature_dim = 768
        self.backbone.train()
    
    def _configure_trainable_layers(self, num_unfrozen_blocks: int):
        """Configure which layers are trainable"""
        # First freeze everything
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        if hasattr(self.backbone, 'blocks'):
            total_blocks = len(self.backbone.blocks)
            num_unfrozen_blocks = min(num_unfrozen_blocks, total_blocks)
            
            # Unfreeze last N blocks
            for i in range(total_blocks - num_unfrozen_blocks, total_blocks):
                for param in self.backbone.blocks[i].parameters():
                    param.requires_grad = True
            
            logger.info(f"Unfrozen last {num_unfrozen_blocks}/{total_blocks} transformer blocks")
            
            # Always unfreeze normalization layers
            if hasattr(self.backbone, 'norm'):
                for param in self.backbone.norm.parameters():
                    param.requires_grad = True
        else:
            # Fallback: unfreeze last few layers
            layers = list(self.backbone.children())
            for layer in layers[-2:]:
                for param in layer.parameters():
                    param.requires_grad = True
            logger.info("Unfrozen last 2 layers (fallback)")
    
    def calculate_alpha(self) -> float:
        """Calculate dynamic alpha for gradient reversal with improved scheduling"""
        if not self.training:
            return 0.0
        
        # Progressive schedule: slower start, stable end
        p = min(float(self.num_updates) / 10000.0, 1.0)
        alpha = 2.0 / (1.0 + np.exp(-8 * p)) - 1.0
        return max(0.0, alpha) * 0.7  # Scale for stability
    
    def mixup_features(self, features: torch.Tensor) -> torch.Tensor:
        """Apply feature-level mixup for regularization"""
        if not self.training or self.mixup_alpha <= 0:
            return features
        
        batch_size = features.size(0)
        if batch_size < 2:
            return features
        
        # Random permutation and lambda
        perm = torch.randperm(batch_size, device=features.device)
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        
        # Apply mixup
        mixed_features = lam * features + (1 - lam) * features[perm]
        return mixed_features
    
    def forward(self, 
                x: torch.Tensor,
                domain_labels: Optional[torch.Tensor] = None,
                return_features: bool = False,
                inference: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        # Calculate alpha for gradient reversal
        alpha = 0.0 if inference else self.calculate_alpha()
        
        # Extract backbone features
        features = self.backbone(x)
        if features.dim() > 2:
            features = features.view(features.size(0), -1)
        
        # Apply domain alignment
        aligned_features = self.domain_alignment(features, domain_labels)
        
        # Apply bottleneck transformation
        bottleneck_features = self.bottleneck(aligned_features)
        
        # Apply feature mixup for regularization
        mixed_features = self.mixup_features(bottleneck_features)
        
        # Task classification
        logits = self.classifier(mixed_features)
        
        # Domain adversarial training
        if self.training and alpha > 0:
            # Add minimal noise for robustness
            noisy_features = bottleneck_features + torch.randn_like(bottleneck_features) * 0.005
            reversed_features = grad_reverse(noisy_features, alpha)
            domain_pred = self.domain_discriminator(reversed_features)
        else:
            domain_pred = self.domain_discriminator(bottleneck_features)
        
        if return_features:
            return logits, domain_pred, bottleneck_features
        
        return logits, domain_pred, bottleneck_features
    
    def get_losses(self, data_dict: Dict, pred_dict: Dict) -> Dict[str, torch.Tensor]:
        """Calculate comprehensive loss with improved weighting"""
        labels = data_dict['label'].long()
        domain_labels = data_dict.get('domain_label', None)
        logits = pred_dict['cls']
        domain_pred = pred_dict['domain_pred']
        
        # Classification loss with label smoothing
        cls_loss = F.cross_entropy(logits, labels, label_smoothing=0.1)
        losses = {'cls': cls_loss}
        
        # For inference, only return classification loss
        if not self.training:
            losses['overall'] = cls_loss
            return losses
        
        # Domain adversarial losses
        if domain_labels is not None:
            domain_labels = domain_labels.long()
            
            # Domain classification loss
            domain_loss = F.cross_entropy(domain_pred, domain_labels)
            losses['domain'] = domain_loss
            
            # Progressive lambda with improved scheduling
            p = min(float(self.num_updates) / 8000.0, 1.0)
            lambda_weight = 2.0 / (1.0 + np.exp(-8 * p)) - 1.0
            lambda_weight = max(0.0, lambda_weight) * 0.3  # More conservative
            
            # Domain confusion loss (encourage domain invariance)
            domain_probs = F.softmax(domain_pred, dim=1)
            uniform_target = torch.ones_like(domain_probs) / self.num_domains
            confusion_loss = F.kl_div(F.log_softmax(domain_pred, dim=1),
                                    uniform_target, reduction='batchmean')
            
            # Entropy regularization
            entropy_loss = torch.tensor(0.0, device=logits.device)
            if self.entropy_conditioning:
                cls_probs = F.softmax(logits, dim=1)
                entropy_loss = -torch.mean(torch.sum(cls_probs * torch.log(cls_probs + 1e-8), dim=1))
            
            # Combine losses with careful weighting
            total_loss = (cls_loss +
                         lambda_weight * domain_loss * 0.3 +
                         confusion_loss * 0.05 +
                         entropy_loss * 0.02)
            
            losses.update({
                'confusion': confusion_loss,
                'entropy': entropy_loss,
                'overall': total_loss,
                'lambda': lambda_weight
            })
        else:
            losses['overall'] = cls_loss
        
        self.num_updates += 1
        return losses
    
    def get_probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        """Get class probabilities from logits"""
        probs = torch.softmax(logits, dim=1)
        if probs.size(1) == 2:
            return probs[:, 1]  # Binary classification: return positive class prob
        return probs
    
    def prepare_batch(self, data_dict: Dict, inference: bool = False) -> Dict[str, torch.Tensor]:
        """Format batch data for benchmark compatibility"""
        logits, domain_pred, features = self(
            data_dict['image'],
            domain_labels=data_dict.get('domain_label', None),
            inference=inference
        )
        
        return {
            'cls': logits,
            'prob': torch.softmax(logits, dim=1),
            'feat': features,
            'domain_pred': domain_pred
        }
    
    def get_train_metrics(self, data_dict: Dict, predictions: Dict) -> Dict[str, float]:
        """Calculate training metrics with enhanced domain-aware logging"""
        logits = predictions['cls']
        domain_pred = predictions['domain_pred']
        labels = data_dict['label'].long()
        
        # Basic classification metrics
        pred_labels = torch.argmax(logits, dim=1)
        acc = (pred_labels == labels).float().mean().item()
        
        probs = torch.softmax(logits, dim=1)
        
        metrics = {
            'acc': acc,
            'probs': logits.detach()
        }
        
        # Probability metrics
        if self.num_classes == 2:
            positive_probs = probs[:, 1]
            metrics['prob_mean'] = positive_probs.mean().item()
        else:
            metrics['prob_mean'] = probs.max(dim=1)[0].mean().item()
        
        # Domain-aware metrics
        if 'domain_label' in data_dict:
            domain_labels = data_dict['domain_label'].long()
            domain_pred_labels = domain_pred.argmax(dim=1)
            domain_acc = (domain_pred_labels == domain_labels).float().mean().item()
            metrics['domain_acc'] = domain_acc
            
            # Per-domain classification accuracy
            domain_accs = []
            for domain_id in range(self.num_domains):
                mask = domain_labels == domain_id
                if mask.sum() > 0:
                    domain_cls_acc = (pred_labels[mask] == labels[mask]).float().mean().item()
                    metrics[f'domain_{domain_id}_cls_acc'] = domain_cls_acc
                    domain_accs.append(domain_cls_acc)
            
            # Periodic detailed logging
            if self.training and self.num_updates % 500 == 0:
                alpha = self.calculate_alpha()
                metrics_str = f"Update {self.num_updates} | Acc: {acc:.4f} | Alpha: {alpha:.3f}"
                if domain_accs:
                    domain_str = ", ".join([f"D{i}: {acc:.3f}" for i, acc in enumerate(domain_accs)])
                    metrics_str += f" | Domain accs: [{domain_str}]"
                logger.info(metrics_str)
        
        return metrics
```

---

# 🧩 ব্যাখ্যা — অংশ ১: **Gradient Reversal Layer (GRL)**

**কোথায়:** `GradientReversalFunction` + `grad_reverse()`
**কাজ:** Forward এ ইনপুট অপরিবর্তিত রাখে, কিন্তু Backward এ গ্র্যাডিয়েন্টকে `-alpha` গুণে **উল্টো** করে দেয়।
**কেন:** ডোমেইন ডিসক্রিমিনেটর ডোমেইন আলাদা করতে চায়; GRL-এর কারণে ফিচার-এক্সট্রাক্টর তার উল্টো দিকে শেখে—ফল: **ডোমেইন-ইনভারিয়্যান্ট ফিচার**।

---

# 🧩 ব্যাখ্যা — অংশ ২: **init\_weights**

**কোথায়:** `init_weights(m)`
**কাজ:** `Linear`-এ Xavier init, `LayerNorm/BatchNorm`-এ ওজন=1, বায়াস=0।
**কেন:** ট্রেনিং স্টেবিলিটি ও গ্র্যাডিয়েন্ট ফ্লো উন্নত করতে—শুরুতেই নেটওয়ার্ককে ভালো অবস্থায় আনা।

---

# 🧩 ব্যাখ্যা — অংশ ৩: **MultiSourceDomainDiscriminator**

**কোথায়:** `class MultiSourceDomainDiscriminator`
**কাজ:** Bottleneck ফিচার থেকে **ডোমেইন প্রেডিকশন** (0..num\_domains-1)।
**আর্কিটেকচার:** `Linear → BN → LeakyReLU → Dropout` ব্লক + final Linear।
**কেন:** একাধিক সোর্স/টার্গেট থাকলে multi-domain হ্যান্ডেল করতে পারে। GRL-এর সাথে adversarially ট্রেন হয়ে ফিচারকে ডোমেইন-ইনভারিয়্যান্ট করতে সাহায্য করে।

---

# 🧩 ব্যাখ্যা — অংশ ৪: **FeatureBottleneck**

**কোথায়:** `class FeatureBottleneck`
**কাজ:** ব্যাকবোন ফিচার → রেগুলারাইজড, কমপ্যাক্ট **বটলনেক ফিচার**।
**কেন:** ওভারফিটিং কমানো, ক্লাসিফায়ারের আগে ফিচার স্পেসকে **শেপ** করা এবং ট্রেনিং স্থিতিশীল রাখা।

---

# 🧩 ব্যাখ্যা — অংশ ৫: **DomainAlignment**

**কোথায়:** `class DomainAlignment`
**কাজ:**

* শেয়ারড ট্রান্সফর্ম (সব ডোমেইনের জন্য সাধারণ ম্যাপিং),
* **ডোমেইন-স্পেসিফিক bias** (হালকা অ্যাডাপ্টেশন),
* শেয়ারড প্রজেকশন (কমন স্পেসে ফিচার)।
  **কেন:** ভারী ডোমেইন-স্পেসিফিক লেয়ার ছাড়াই কম প্যারামিটারে **এফিসিয়েন্ট অ্যালাইনমেন্ট**।

---

# 🧩 ব্যাখ্যা — অংশ ৬: **asif\_clip\_dan — কন্সট্রাকশন**

**কোথায়:** `class asif_clip_dan`, `__init__`, `from_config`
**কাজ:**

* **Backbone**: CLIP ViT-B/32 ট্রাই; না হলে স্ট্যান্ডার্ড ViT। Head = `Identity`।
* **feature\_dim** অটো-ডিটেক্ট (ডামি ফরওয়ার্ড করে)।
* **Last N blocks** আনফ্রিজ (ডিফল্ট 6) + সব `norm` ট্রেনেবল—প্র্যাকটিকাল ফাইন-টিউনিং।
* মডিউল জুড়ে: `DomainAlignment` + `FeatureBottleneck` + `Classifier` + `DomainDiscriminator`।

---

# 🧩 ব্যাখ্যা — অংশ ৭: **calculate\_alpha (Dynamic GRL strength)**

**কোথায়:** `calculate_alpha()`
**কাজ:** ট্রেনিং আপডেটের উপর ভিত্তি করে `0 → ~0.7` পর্যন্ত ধীরে ধীরে বাড়ে।
**কেন:** শুরুতে টাস্ক শেখা জোরালো, ধীরে ধীরে ডোমেইন-ইনভারিয়্যান্স টাইট—**স্থিতিশীল ট্রেনিং**।

---

# 🧩 ব্যাখ্যা — অংশ ৮: **mixup\_features**

**কোথায়:** `mixup_features()`
**কাজ:** ফিচার-লেভেলে mixup (β-বিতরণ থেকে λ), ব্যাচ পারমিউটেড জোড়া দিয়ে মিক্স।
**কেন:** ডেসিশন বাউন্ডারি স্মুথ করা, জেনারালাইজেশন বাড়ানো, ওভারফিটিং কমানো।

---

# 🧩 ব্যাখ্যা — অংশ ৯: **forward**

**প্রবাহ:**

1. `backbone(x)` → raw features
2. `domain_alignment(features, domain_labels)`
3. `bottleneck(...)`
4. `mixup(...)` (train হলে)
5. `classifier(...)` → **logits**
6. **domain\_pred**:

   * train & α>0 হলে: bottleneck + ছোট noise → **GRL(α)** → discriminator
   * না হলে: bottleneck → discriminator
     **আউটপুট:** `logits, domain_pred, bottleneck_features` (features চাইলে রিটার্ন হয়)

---

# 🧩 ব্যাখ্যা — অংশ ১০: **get\_losses**

**কাজ:** সব লস গণনা ও ওজনায়ন।

* **cls\_loss:** CE + label\_smoothing(0.1)
* (train+domain\_labels থাকলে)

  * **domain\_loss:** CE(domain\_pred, domain\_labels)
  * **lambda\_weight:** আপডেট-ভিত্তিক স্কেডিউল, কনজারভেটিভ (*×0.3*)
  * **confusion\_loss:** KL-div → ডিসক্রিমিনেটর আউটপুটকে ইউনিফর্মের দিকে টানে (ডোমেইন indistinguishable)
  * **entropy\_loss:** ক্লাস প্রোব-এন্ট্রপি (over-confidence কমাতে)
* **overall:**
  `cls + lambda*domain*0.3 + confusion*0.05 + entropy*0.02`
* শেষে `self.num_updates += 1` (সিডিউলার আপডেট)

---

# 🧩 ব্যাখ্যা — অংশ ১১: **get\_probabilities**

**কাজ:** logits → softmax probs; বাইনারি হলে পজিটিভ ক্লাসের প্রোব রিটার্ন।
**উপযোগিতা:** ROC/AUC, থ্রেশহোল্ডিং ইত্যাদি।

---

# 🧩 ব্যাখ্যা — অংশ ১২: **prepare\_batch**

**কাজ:** এক কলেই `forward` চালিয়ে ডিকশনারি বানায়:
`{'cls', 'prob', 'feat', 'domain_pred'}`
**উপযোগিতা:** ট্রেন লুপে সরাসরি ব্যবহারযোগ্য একীভূত ফরম্যাট।

---

# 🧩 ব্যাখ্যা — অংশ ১৩: **get\_train\_metrics**

**কাজ:**

* **acc** (ক্লাসিফিকেশন), **prob\_mean**
* যদি domain\_label থাকে: **domain\_acc** + **per-domain cls acc**
* প্রতি 500 আপডেটে লগে **alpha** ও ডোমেইনভিত্তিক এক্যুরেসি প্রিন্ট—ডিবাগ/মনিটরিংয়ে সহায়ক।

---



---

## 🧩 Part 1 — Code (GRL)

```python
import torch
import torch.nn as nn

class GradientReversalFunction(torch.autograd.Function):
    """Gradient Reversal Layer for domain adversarial training"""
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None

def grad_reverse(x, alpha=1.0):
    """Apply gradient reversal with given alpha"""
    return GradientReversalFunction.apply(x, alpha)
```

---

## 🔎 Line-by-line ব্যাখ্যা

**`import torch`**

* PyTorch লাইব্রেরি ইমপোর্ট করা হচ্ছে—টেনসর, অটো-গ্র্যাড, অপারেশন ইত্যাদির জন্য।

**`import torch.nn as nn`**

* `torch.nn` শর্টনাম `nn` হিসেবে ইমপোর্ট—নিউরাল নেটওয়ার্ক কম্পোনেন্ট বানানোর সুবিধার জন্য (যদিও এই টুকুর ভেতরে সরাসরি `nn` ইউজ করিনি, পরে কাজে লাগতে পারে)।

---

**`class GradientReversalFunction(torch.autograd.Function):`**

* PyTorch-এর **custom autograd Function** ডিফাইন করছি—এখানে আমরা ব্যাকওয়ার্ড পাস কাস্টমাইজ করব, যাতে গ্র্যাডিয়েন্ট **উল্টো** দিকে যায়।

**`"""Gradient Reversal Layer for domain adversarial training"""`**

* ডকস্ট্রিং: এই ক্লাসটা ডোমেইন-অ্যাডভার্সারিয়াল ট্রেনিংয়ের GRL হিসেবে কাজ করবে।

---

**`@staticmethod`**

* `forward`/`backward` দুটোই স্ট্যাটিক মেথড—ইনস্ট্যান্স ছাড়াই ফ্রেমওয়ার্ক কল করবে।

**`def forward(ctx, x, alpha):`**

* **Forward-pass** মেথড।
* `ctx`: কনটেক্সট অবজেক্ট—এখানে এমন ভ্যারিয়েবল রাখা হবে যা ব্যাকওয়ার্ডে লাগবে।
* `x`: ইনপুট ফিচার টেনসর।
* `alpha`: GRL-এর শক্তি (λ/α)—কতটা উল্টো গ্র্যাডিয়েন্ট লাগবে।

**`ctx.alpha = alpha`**

* ব্যাকওয়ার্ড পাসে ব্যবহার করার জন্য `alpha` কনটেক্সটে সংরক্ষণ করা হচ্ছে।

**`return x.view_as(x)`**

* ফরওয়ার্ডে **কোনো পরিবর্তন করা হচ্ছে না**—ইনপুট যেমন আছে তেমনই রিটার্ন (shape-preserving ভিউ দিয়ে)।
* মানে GRL forward = **Identity**।

---

**`@staticmethod`**

* ব্যাকওয়ার্ড মেথডও স্ট্যাটিক।

**`def backward(ctx, grad_output):`**

* **Backward-pass**: আপস্ট্রিম থেকে আসা গ্র্যাডিয়েন্ট `grad_output` পাওয়া যাচ্ছে।

**`return -ctx.alpha * grad_output, None`**

* মূল কাজ: গ্র্যাডিয়েন্টকে **উল্টো** করে দেওয়া → `-alpha * grad_output`।
* প্রথম রিটার্ন ইনপুট `x`-এর গ্র্যাডিয়েন্ট;
* দ্বিতীয় রিটার্ন `alpha`-র জন্য গ্র্যাডিয়েন্ট (প্রয়োজন নেই) তাই `None`।
* এর ফলে ফিচার এক্সট্রাক্টর **ডোমেইন ডিসক্রিমিনেটরের বিপরীত** দিকে অপটিমাইজ হবে → **ডোমেইন-ইনভারিয়্যান্ট ফিচার**।

---

**`def grad_reverse(x, alpha=1.0):`**

* ছোট হেল্পার ফাংশন—ব্যবহার সহজ করার জন্য।

**`"""Apply gradient reversal with given alpha"""`**

* ডকস্ট্রিং: নির্দিষ্ট `alpha` দিয়ে GRL অ্যাপ্লাই করা হবে।

**`return GradientReversalFunction.apply(x, alpha)`**

* কাস্টম autograd ফাংশনের `.apply()` কল—পর্দার আড়ালে `forward`/`backward` ঠিকমতো হ্যান্ডল হবে।
* আপনি যেখানেই `features`-এর ওপর GRL লাগাতে চান, সেখানে `grad_reverse(features, alpha)` লিখলেই হবে।

---

### 🧠 সংক্ষিপ্ত সারমর্ম

* **Forward:** ডাটা বদলায় না (identity)।
* **Backward:** গ্র্যাডিয়েন্ট `-alpha` গুণে উল্টো হয়।
* **ফলাফল:** ফিচার এক্সট্রাক্টর ডোমেইন আলাদা করা **কঠিন** করে এমন ফিচার শেখে → **domain-invariant** representation।

---

ঠিক আছে—এই `GradientReversalFunction` আর `grad_reverse` টুকুর কাজ একদম সহজ ভাষায়, ধাপে ধাপে ব্যাখ্যা করছি।

---

## এরা কী করে?

Gradient Reversal Layer (GRL) এমন একটি “লেয়ার” যা **forward-pass**-এ ইনপুটকে *একদমই বদলায় না*, কিন্তু **backward-pass**-এ এর গ্র্যাডিয়েন্টের সাইন **উল্টো করে দেয়** (এবং চাইলে মাত্রা `alpha` দিয়ে স্কেল করে)।
ডোমেইন-অ্যাডভার্সেরিয়াল ট্রেনিং (DANN)-এ এটা ব্যবহার করা হয় যাতে ফিচার এক্সট্রাক্টর ডোমেইন-ইনভারিয়্যান্ট ফিচার শিখে।

---

## কোডটা কীভাবে কাজ করে?

```python
class GradientReversalFunction(torch.autograd.Function):
    """Gradient Reversal Layer for domain adversarial training"""
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None
```

### 1) `torch.autograd.Function`

* PyTorch-এ কাস্টম autograd অপারেটর বানাতে `torch.autograd.Function` সাবক্লাস করতে হয়।
* এখানে দু’টি স্ট্যাটিক মেথড থাকে:

  * `forward(ctx, ...)`: ফরোয়ার্ড কম্পিউটেশনের সময় কী হবে।
  * `backward(ctx, grad_output)`: ব্যাকওয়ার্ডে আগের লেয়ারের দিকে কী গ্র্যাডিয়েন্ট ফেরত যাবে।

### 2) `forward(ctx, x, alpha)`

* `ctx` হলো context অবজেক্ট—এখানে আপনি এমন তথ্য রেখে দিতে পারেন যেটা পরে `backward`-এ লাগবে।
* `ctx.alpha = alpha`: আমরা `alpha` রেখে দিচ্ছি যাতে পরে গ্র্যাডিয়েন্ট স্কেল করতে পারি।
* `return x.view_as(x)`: ইনপুট `x`-কেই অপরিবর্তিত ফিরিয়ে দিচ্ছে। (মানে forward-এ GRL কিছুই বদলায় না।)

### 3) `backward(ctx, grad_output)`

* `grad_output` হলো এই অপারেটরের আউটপুটের ওপর লসের গ্র্যাডিয়েন্ট, যা autograd দিয়ে এসেছে।
* আমরা রিটার্ন করছি `-ctx.alpha * grad_output` — মানে গ্র্যাডিয়েন্টের সাইন উল্টো (নেগেটিভ) করে এবং `alpha` দিয়ে স্কেল করে আগের লেয়ারে পাঠাচ্ছি।
* দ্বিতীয় `None` টা `alpha`-র গ্র্যাডিয়েন্ট; এখানে `alpha` শেখানো হচ্ছে না (কনস্ট্যান্ট হিসেবে ধরা), তাই `None`।

---

## হেল্পার ফাংশন: `grad_reverse`

```python
def grad_reverse(x, alpha=1.0):
    """Apply gradient reversal with given alpha"""
    return GradientReversalFunction.apply(x, alpha)
```

* ব্যবহার সহজ করার জন্য একটা র‍্যাপার—এখান থেকে `GradientReversalFunction.apply` কল করলেই হলো।
* `alpha` না দিলে ডিফল্ট 1.0 (মানে গ্র্যাডিয়েন্ট কেবল সাইন উল্টো হবে)।

---

## ছোট্ট ইন্টুইশন (কেন দরকার?)

* ধরুন আপনার একটি **feature extractor F** ও তার ওপর একটি **domain discriminator D** আছে।
* আপনি চান: **ক্লাসিফিকেশন ঠিক থাকুক**, কিন্তু **ডিসক্রিমিনেটর যেন ডোমেইন ধরতে না পারে**—তাহলে ফিচারগুলো ডোমেইন-ইনভারিয়্যান্ট হবে।
* D-কে ট্রেন করতে গেলে D-এর লস কমাতে হবে → এর জন্য **ফিচারে D-এর গ্র্যাডিয়েন্ট** নরমালি ব্যাকপ্রপে আসে।
* কিন্তু F-কে “ডোমেইন-ফুল” না করে “ডোমেইন-ইনভারিয়্যান্ট” করতে চাইলে, **F-এর দিকে D-এর গ্র্যাডিয়েন্টের সাইন উল্টো** পাঠাই (GRL)।
  ফলে D ভালো হতে চাইলে F উল্টো দিকের আপডেট পায়—এই অ্যাডভার্সেরিয়াল টানাপোড়েনে ফিচার ডোমেইন-ইনভারিয়্যান্ট হয়।

---

## ছোট্ট কোড উদাহরণ (ব্যবহার)

```python
# features: extractor থেকে আসা ফিচার
# domain_disc: ডোমেইন ডিসক্রিমিনেটর (একটা MLP ধরা যাক)
alpha = 0.5  # ট্রেনিং স্টেপের সাথে ধীরে ধীরে বাড়াতে পারেন

# Forward (GRL forward কিছুই বদলায় না)
rev_features = grad_reverse(features, alpha)

# Domain logits
domain_logits = domain_disc(rev_features)

# Domain loss
domain_labels = torch.randint(0, num_domains, (features.size(0),), device=features.device)
domain_loss = F.cross_entropy(domain_logits, domain_labels)

# এই domain_loss ব্যাকপ্রপালে গেলে GRL ব্যাকওয়ার্ডে গ্র্যাডিয়েন্টকে -alpha দিয়ে উল্টো দেবে
domain_loss.backward()
```

---

## সাধারণ ভুল/টিপস

* **`alpha` খুব বড়** দিলে ট্রেনিং অস্থির হতে পারে—ধীরে ধীরে বাড়ানোর (scheduling) অভ্যাস ভালো।
* GRL **forward-এ কিছু পরিবর্তন করে না**—অনেকে ভাবেন এটা ফিচারও বদলায়; না, শুধু backward-এ প্রভাব ফেলে।
* `alpha`-র গ্র্যাড এখানে **None**—মানে `alpha` শিখছে না; আপনি চাইলে নিজে স্কেজ্যুল করে সেট করবেন।


