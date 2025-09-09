import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(torch.nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        intersection = (pred * target).sum()
        denominator = pred.sum() + target.sum()
        dice_score = (2. * intersection + self.smooth) / (denominator + self.smooth)
        dice_loss = 1. - dice_score
        return dice_loss

class clDiceLoss(torch.nn.Module):
    def __init__(self, smooth=1e-6):
        super(clDiceLoss, self).__init__()
        self.smooth = smooth

    def soft_cldice_loss(self, pred, target, target_skeleton=None):
        '''
        inputs shape  (batch, channel, height, width).
        calculate clDice loss
        Because pred and target at moment of loss calculation will be a torch tensors
        it is preferable to calculate target_skeleton on the step of batch forming,
        when it will be in numpy array format by means of opencv
        '''
        cl_pred = self.soft_skeletonize(pred)
        if target_skeleton is None:
            target_skeleton = self.soft_skeletonize(target)
        iflat = self.norm_intersection(cl_pred, target)
        tflat = self.norm_intersection(target_skeleton, pred)
        intersection = (iflat * tflat).sum()
        return 1. - (2. * intersection) / (iflat + tflat).sum()

    def dice_loss(self, pred, target):
        '''
        inputs shape  (batch, channel, height, width).
        calculate dice loss per batch and channel of sample.
        E.g. if batch shape is [64, 1, 128, 128] -> [64, 1]
        '''
        intersection = (pred * target).sum()
        denominator = pred.sum() + target.sum()
        dice_score = (2. * intersection + self.smooth) / (denominator + self.smooth)
        dice_loss = 1. - dice_score
        return dice_loss

    def soft_skeletonize(self, x, thresh_width=10):
        '''
        Differenciable aproximation of morphological skelitonization operaton
        thresh_width - maximal expected width of vessel
        '''
        for i in range(thresh_width):
            min_pool_x = torch.nn.functional.max_pool2d(x * -1, (3, 3), 1, 1) * -1
            contour = torch.nn.functional.relu(torch.nn.functional.max_pool2d(min_pool_x, (3, 3), 1, 1) - min_pool_x)
            x = torch.nn.functional.relu(x - contour)
        return x

    def norm_intersection(self, center_line, vessel):
        '''
        inputs shape  (batch, channel, height, width)
        intersection formalized by first ares
        x - suppose to be centerline of vessel (pred or gt) and y - is vessel (pred or gt)
        '''
        smooth = 1.
        clf = center_line.view(*center_line.shape[:2], -1)
        vf = vessel.view(*vessel.shape[:2], -1)
        intersection = (clf * vf).sum(-1)
        return (intersection + smooth) / (clf.sum(-1) + smooth)

    def forward(self, pred, target):
        return 0.8 * self.dice_loss(pred, target) + 0.2 * self.soft_cldice_loss(pred, target)

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class MCELoss(torch.nn.Module):
    def __init__(self, reduction='mean'):
        super(MCELoss, self).__init__()
        self.reduction = reduction

    def forward(self, inputs, targets):
        predicted_classes = inputs.argmax(dim=1)
        incorrect_predictions = predicted_classes != targets
        mce_loss = incorrect_predictions.float().mean()

        if self.reduction == 'mean':
            return mce_loss
        elif self.reduction == 'sum':
            return mce_loss * inputs.size(0)
        else:
            return mce_loss

class FocalDiceLoss(torch.nn.Module):
    """Focal Dice Loss - 针对困难样本的改进"""
    
    def __init__(self, alpha=0.25, gamma=2.0, smooth=1e-6):
        super(FocalDiceLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
    
    def forward(self, pred, target):
        # 计算基础dice score
        intersection = (pred * target).sum()
        denominator = pred.sum() + target.sum()
        dice_score = (2. * intersection + self.smooth) / (denominator + self.smooth)
        dice_loss = 1. - dice_score
        
        # Focal权重：困难样本(dice_score低)获得更高权重
        focal_weight = self.alpha * (dice_loss ** self.gamma)
        
        return focal_weight * dice_loss

class BoundaryLoss(torch.nn.Module):
    """边界损失 - 加强对血管边缘的识别"""
    
    def __init__(self, smooth=1e-6):
        super(BoundaryLoss, self).__init__()
        self.smooth = smooth
    
    def get_boundary(self, x):
        """提取边界 - 使用形态学膨胀和腐蚀的差值"""
        # 定义膨胀和腐蚀的卷积核
        kernel_size = 3
        
        # 膨胀操作
        dilated = F.max_pool2d(x, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        
        # 腐蚀操作
        eroded = -F.max_pool2d(-x, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        
        # 边界 = 膨胀 - 腐蚀
        boundary = torch.abs(dilated - eroded)
        
        return boundary
    
    def forward(self, pred, target):
        # 提取预测和目标的边界
        pred_boundary = self.get_boundary(pred)
        target_boundary = self.get_boundary(target)
        
        # 计算边界损失 (使用L2损失)
        boundary_loss = F.mse_loss(pred_boundary, target_boundary)
        
        return boundary_loss

class ConservativeEnhancedLoss(torch.nn.Module):
    """保守增强损失 - 在原有基础上只加入5%的focal权重"""
    
    def __init__(self, smooth=1e-6):
        super(ConservativeEnhancedLoss, self).__init__()
        self.smooth = smooth
        
        # 保持原有损失函数
        self.dice_loss = DiceLoss(smooth=smooth)
        self.cldice_loss = clDiceLoss(smooth=smooth)
        self.focal_dice_loss = FocalDiceLoss(smooth=smooth)
        
        # 保守权重：原有0.8:0.2减少5%，新增5%focal
        self.dice_weight = 0.76     # 0.8 - 0.04
        self.cldice_weight = 0.19   # 0.2 - 0.01  
        self.focal_weight = 0.05    # 新增focal权重
        
        print(f"保守增强损失权重: Dice={self.dice_weight:.2f}, clDice={self.cldice_weight:.2f}, Focal={self.focal_weight:.2f}")
    
    def forward(self, pred, target):
        # 计算各损失分量
        dice = self.dice_loss(pred, target)
        cldice = self.cldice_loss.soft_cldice_loss(pred, target)
        focal = self.focal_dice_loss(pred, target)
        
        # 加权组合
        total_loss = (self.dice_weight * dice + 
                      self.cldice_weight * cldice + 
                      self.focal_weight * focal)
        
        return total_loss

class EnhancedBoundaryLoss(torch.nn.Module):
    """增强边界损失 - 在原有损失基础上添加0.1权重的边界损失"""
    
    def __init__(self, smooth=1e-6):
        super(EnhancedBoundaryLoss, self).__init__()
        self.smooth = smooth
        
        # 基础损失组件
        self.dice_loss = DiceLoss(smooth=smooth)
        self.cldice_loss = clDiceLoss(smooth=smooth)
        self.boundary_loss = BoundaryLoss(smooth=smooth)
        
        # 权重分配: 原始权重减少10%，分配给边界损失
        self.dice_weight = 0.72     # 原来0.8，减少0.08
        self.cldice_weight = 0.18   # 原来0.2，减少0.02
        self.boundary_weight = 0.10  # 新增10%边界损失权重
        
        print(f"增强边界损失权重: Dice={self.dice_weight:.2f}, clDice={self.cldice_weight:.2f}, Boundary={self.boundary_weight:.2f}")
    
    def forward(self, pred, target):
        # 计算各损失分量
        dice = self.dice_loss(pred, target)
        cldice = self.cldice_loss.soft_cldice_loss(pred, target)
        boundary = self.boundary_loss(pred, target)
        
        # 加权组合
        total_loss = (self.dice_weight * dice + 
                      self.cldice_weight * cldice + 
                      self.boundary_weight * boundary)
        
        return total_loss

class MinimalEnhancedLoss(nn.Module):
    """最小化改动的增强损失函数 - 仅添加1%的Focal权重"""
    def __init__(self):
        super(MinimalEnhancedLoss, self).__init__()
        self.dice_loss = DiceLoss()
        self.cldice_loss = clDiceLoss()
        self.focal_loss = FocalDiceLoss()
        
        # 最保守的权重分配：99%保持原有，1%新增
        self.dice_weight = 0.79    # 原来0.8
        self.cldice_weight = 0.20  # 保持0.2
        self.focal_weight = 0.01   # 新增1%
        
    def forward(self, pred, target):
        dice = self.dice_loss(pred, target)
        cldice = self.cldice_loss(pred, target)  
        focal = self.focal_loss(pred, target)
        
        total_loss = (self.dice_weight * dice + 
                     self.cldice_weight * cldice + 
                     self.focal_weight * focal)
        
        return total_loss

class UltraConservativeLoss(nn.Module):
    """超保守损失函数 - 仅添加0.5%的Focal权重"""
    def __init__(self):
        super(UltraConservativeLoss, self).__init__()
        self.dice_loss = DiceLoss()
        self.cldice_loss = clDiceLoss()
        self.focal_loss = FocalDiceLoss()
        
        # 超保守权重：99.5%保持原有，0.5%新增
        self.dice_weight = 0.796   # 原来0.8，减少0.004
        self.cldice_weight = 0.199 # 原来0.2，减少0.001
        self.focal_weight = 0.005  # 新增0.5%
        
    def forward(self, pred, target):
        dice = self.dice_loss(pred, target)
        cldice = self.cldice_loss(pred, target)  
        focal = self.focal_loss(pred, target)
        
        total_loss = (self.dice_weight * dice + 
                     self.cldice_weight * cldice + 
                     self.focal_weight * focal)
        
        return total_loss

# =================== 医学图像分割常用损失函数 ===================

class CrossEntropyLoss(nn.Module):
    """标准交叉熵损失 - 医学分割基础损失"""
    def __init__(self, weight=None, ignore_index=-100, reduction='mean'):
        super(CrossEntropyLoss, self).__init__()
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction
        
    def forward(self, pred, target):
        # pred: [B, C, H, W], target: [B, H, W] 或 [B, 1, H, W]
        if target.dim() == 4 and target.size(1) == 1:
            target = target.squeeze(1)
        if target.dim() == 4:
            target = target.argmax(dim=1)
            
        return F.cross_entropy(pred, target.long(), 
                             weight=self.weight, 
                             ignore_index=self.ignore_index,
                             reduction=self.reduction)

class WeightedCrossEntropyLoss(nn.Module):
    """加权交叉熵损失 - 处理类别不平衡"""
    def __init__(self, class_weights=None, reduction='mean'):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.class_weights = class_weights
        self.reduction = reduction
        
    def forward(self, pred, target):
        # 自动计算类别权重
        if self.class_weights is None:
            # 计算类别频率的倒数作为权重
            unique, counts = torch.unique(target, return_counts=True)
            total = target.numel()
            weights = total / (len(unique) * counts.float())
            self.class_weights = weights.to(pred.device)
        
        return F.cross_entropy(pred, target.long(), 
                             weight=self.class_weights,
                             reduction=self.reduction)

class TverskyLoss(nn.Module):
    """Tversky损失 - 可调节FP和FN的权重"""
    def __init__(self, alpha=0.3, beta=0.7, smooth=1e-6):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha  # FP权重
        self.beta = beta    # FN权重  
        self.smooth = smooth
        
    def forward(self, pred, target):
        # 计算True Positive, False Positive, False Negative
        TP = (pred * target).sum()
        FP = (pred * (1 - target)).sum()
        FN = ((1 - pred) * target).sum()
        
        tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        
        return 1 - tversky

class FocalTverskyLoss(nn.Module):
    """Focal Tversky损失 - 结合Focal和Tversky的优势"""
    def __init__(self, alpha=0.3, beta=0.7, gamma=2.0, smooth=1e-6):
        super(FocalTverskyLoss, self).__init__()
        self.tversky_loss = TverskyLoss(alpha, beta, smooth)
        self.gamma = gamma
        
    def forward(self, pred, target):
        tversky_loss = self.tversky_loss(pred, target)
        focal_tversky = torch.pow(tversky_loss, self.gamma)
        
        return focal_tversky

class IoULoss(nn.Module):
    """IoU损失 - 交并比损失"""
    def __init__(self, smooth=1e-6):
        super(IoULoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, pred, target):
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum() - intersection
        
        iou = (intersection + self.smooth) / (union + self.smooth)
        
        return 1 - iou

class SSLoss(nn.Module):
    """Sensitivity-Specificity损失"""
    def __init__(self, r=0.1, smooth=1e-6):
        super(SSLoss, self).__init__()
        self.r = r  # sensitivity和specificity的权重平衡
        self.smooth = smooth
        
    def forward(self, pred, target):
        # Sensitivity (Recall): TP / (TP + FN)
        TP = (pred * target).sum()
        FN = ((1 - pred) * target).sum()
        sensitivity = (TP + self.smooth) / (TP + FN + self.smooth)
        
        # Specificity: TN / (TN + FP)  
        TN = ((1 - pred) * (1 - target)).sum()
        FP = (pred * (1 - target)).sum()
        specificity = (TN + self.smooth) / (TN + FP + self.smooth)
        
        # SS损失
        ss_loss = self.r * (1 - sensitivity) + (1 - self.r) * (1 - specificity)
        
        return ss_loss

class UnifiedFocalLoss(nn.Module):
    """统一Focal损失 - 结合分割和检测的Focal思想"""
    def __init__(self, weight=1.0, delta=0.6, gamma=2.0):
        super(UnifiedFocalLoss, self).__init__()
        self.weight = weight
        self.delta = delta
        self.gamma = gamma
        
    def forward(self, pred, target):
        # 计算基础交叉熵
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        
        # 计算概率
        prob = torch.sigmoid(pred)
        
        # Asymmetric focusing
        prob_t = prob * target + (1 - prob) * (1 - target)
        
        # Modulating factor
        modulating_factor = torch.abs(target - prob)
        modulating_factor = torch.pow(modulating_factor, self.gamma)
        
        # Unified focal loss
        focal_loss = modulating_factor * bce
        
        return self.weight * focal_loss.mean()

class ComboLoss(nn.Module):
    """组合损失 - 结合多种损失的优势"""
    def __init__(self, alpha=0.5, ce_ratio=0.5, smooth=1e-6):
        super(ComboLoss, self).__init__()
        self.alpha = alpha      # Dice和CE的平衡
        self.ce_ratio = ce_ratio
        self.smooth = smooth
        
    def forward(self, pred, target):
        # Dice部分
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        dice_loss = 1 - dice
        
        # CE部分
        bce = F.binary_cross_entropy_with_logits(pred, target)
        
        # 组合
        combo = (self.ce_ratio * bce) + ((1 - self.ce_ratio) * dice_loss)
        
        return combo

class LovaszSoftmaxLoss(nn.Module):
    """Lovasz-Softmax损失 - 直接优化IoU"""
    def __init__(self, smooth=1e-6):
        super(LovaszSoftmaxLoss, self).__init__()
        self.smooth = smooth
        
    def lovasz_grad(self, gt_sorted):
        """计算Lovasz梯度"""
        p = len(gt_sorted)
        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.float().cumsum(0)
        union = gts + (1 - gt_sorted).float().cumsum(0)
        jaccard = 1. - intersection / union
        if p > 1:
            jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
        return jaccard
        
    def forward(self, pred, target):
        # Flatten
        pred = pred.view(-1)
        target = target.view(-1)
        
        # Sort
        errors = (pred - target).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        gt_sorted = target[perm]
        
        # Lovasz梯度
        grad = self.lovasz_grad(gt_sorted)
        loss = torch.dot(F.relu(errors_sorted), grad)
        
        return loss

class HausdorffLoss(nn.Module):
    """Hausdorff距离损失 - 关注边界质量"""
    def __init__(self, alpha=2.0, smooth=1e-6):
        super(HausdorffLoss, self).__init__()
        self.alpha = alpha
        self.smooth = smooth
        
    def compute_dtm(self, img_gt, out_shape):
        """计算距离变换图"""
        import scipy.ndimage as ndi
        
        fg_dtm = ndi.distance_transform_edt(img_gt)
        bg_dtm = ndi.distance_transform_edt(1 - img_gt)
        
        dtm = fg_dtm + bg_dtm
        return torch.tensor(dtm, dtype=torch.float32)
        
    def forward(self, pred, target):
        # 简化版本：使用L2距离近似Hausdorff距离
        pred_edge = self.get_edge(pred)
        target_edge = self.get_edge(target)
        
        # 计算边界之间的距离
        hausdorff_loss = F.mse_loss(pred_edge, target_edge)
        
        return hausdorff_loss
        
    def get_edge(self, x):
        """提取边界"""
        # 使用Sobel算子提取边界
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
        
        edge_x = F.conv2d(x, sobel_x, padding=1)
        edge_y = F.conv2d(x, sobel_y, padding=1)
        
        edge = torch.sqrt(edge_x**2 + edge_y**2 + 1e-6)
        
        return edge

class AsymmetricLoss(nn.Module):
    """非对称损失 - 对FP和FN给予不同惩罚"""
    def __init__(self, beta_pos=1.0, beta_neg=4.0, gamma_pos=2.0, gamma_neg=1.0):
        super(AsymmetricLoss, self).__init__()
        self.beta_pos = beta_pos
        self.beta_neg = beta_neg  
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        
    def forward(self, pred, target):
        # 正样本和负样本的asymmetric focusing
        prob = torch.sigmoid(pred)
        
        # 正样本损失
        pos_loss = target * torch.pow(1 - prob, self.gamma_pos) * F.logsigmoid(pred)
        
        # 负样本损失  
        neg_loss = (1 - target) * torch.pow(prob, self.gamma_neg) * F.logsigmoid(-pred)
        
        # 加权组合
        loss = -(self.beta_pos * pos_loss + self.beta_neg * neg_loss)
        
        return loss.mean()

# =================== 血管分割专用损失函数 ===================

class VesselSegmentationLoss(nn.Module):
    """血管分割专用损失函数 - 针对管状结构优化"""
    
    def __init__(self, 
                 dice_weight=0.3,      # Dice损失权重
                 cldice_weight=0.4,    # clDice损失权重(重点)  
                 boundary_weight=0.2,  # 边界损失权重
                 focal_weight=0.1,     # Focal损失权重
                 smooth=1e-6):
        super(VesselSegmentationLoss, self).__init__()
        
        # 损失函数组件
        self.dice_loss = DiceLoss(smooth=smooth)
        self.cldice_loss = clDiceLoss(smooth=smooth) 
        self.boundary_loss = BoundaryLoss(smooth=smooth)
        self.focal_loss = FocalDiceLoss(alpha=0.25, gamma=2.0, smooth=smooth)
        
        # 权重配置
        self.dice_weight = dice_weight
        self.cldice_weight = cldice_weight      # clDice最重要，管状结构连通性
        self.boundary_weight = boundary_weight  # 边界细节很重要
        self.focal_weight = focal_weight        # 处理困难样本
        
        # 验证权重和为1
        total_weight = dice_weight + cldice_weight + boundary_weight + focal_weight
        if abs(total_weight - 1.0) > 1e-6:
            print(f"⚠️  权重和不为1.0: {total_weight:.3f}")
        
        print(f"🩸 血管分割损失函数配置:")
        print(f"   • Dice损失权重:     {self.dice_weight:.1f}")
        print(f"   • clDice损失权重:    {self.cldice_weight:.1f} (管状连通性)")
        print(f"   • 边界损失权重:     {self.boundary_weight:.1f}")
        print(f"   • Focal损失权重:    {self.focal_weight:.1f}")
        
    def forward(self, pred, target):
        # 计算各个损失分量
        dice = self.dice_loss(pred, target)
        cldice = self.cldice_loss(pred, target)
        boundary = self.boundary_loss(pred, target)  
        focal = self.focal_loss(pred, target)
        
        # 加权组合
        total_loss = (self.dice_weight * dice + 
                     self.cldice_weight * cldice +
                     self.boundary_weight * boundary +
                     self.focal_weight * focal)
        
        return total_loss
    
    def get_loss_details(self, pred, target):
        """返回各个损失分量的详细信息，用于调试"""
        with torch.no_grad():
            dice = self.dice_loss(pred, target)
            cldice = self.cldice_loss(pred, target)
            boundary = self.boundary_loss(pred, target)
            focal = self.focal_loss(pred, target)
            
            total = (self.dice_weight * dice + 
                    self.cldice_weight * cldice +
                    self.boundary_weight * boundary +
                    self.focal_weight * focal)
            
            return {
                'total_loss': total.item(),
                'dice': dice.item(),
                'cldice': cldice.item(), 
                'boundary': boundary.item(),
                'focal': focal.item(),
                'weighted_dice': (self.dice_weight * dice).item(),
                'weighted_cldice': (self.cldice_weight * cldice).item(),
                'weighted_boundary': (self.boundary_weight * boundary).item(),
                'weighted_focal': (self.focal_weight * focal).item()
            }

class EnhancedVesselLoss(nn.Module):
    """增强版血管分割损失 - 添加Tversky和SS损失"""
    
    def __init__(self, 
                 dice_weight=0.25,     # Dice损失权重
                 cldice_weight=0.35,   # clDice损失权重(最重要)
                 boundary_weight=0.15, # 边界损失权重  
                 tversky_weight=0.15,  # Tversky损失权重(处理FP/FN不平衡)
                 ss_weight=0.1,        # SS损失权重(敏感性/特异性平衡)
                 smooth=1e-6):
        super(EnhancedVesselLoss, self).__init__()
        
        # 损失函数组件
        self.dice_loss = DiceLoss(smooth=smooth)
        self.cldice_loss = clDiceLoss(smooth=smooth)
        self.boundary_loss = BoundaryLoss(smooth=smooth)
        self.tversky_loss = TverskyLoss(alpha=0.3, beta=0.7, smooth=smooth)  # 更关注FN
        self.ss_loss = SSLoss(r=0.1, smooth=smooth)  # 平衡敏感性和特异性
        
        # 权重配置
        self.dice_weight = dice_weight
        self.cldice_weight = cldice_weight
        self.boundary_weight = boundary_weight
        self.tversky_weight = tversky_weight
        self.ss_weight = ss_weight
        
        print(f"🩸 增强版血管分割损失函数:")
        print(f"   • Dice:     {self.dice_weight:.2f}")
        print(f"   • clDice:   {self.cldice_weight:.2f} (连通性)")
        print(f"   • Boundary: {self.boundary_weight:.2f} (边界)")
        print(f"   • Tversky:  {self.tversky_weight:.2f} (FP/FN平衡)")
        print(f"   • SS:       {self.ss_weight:.2f} (敏感性/特异性)")
        
    def forward(self, pred, target):
        dice = self.dice_loss(pred, target)
        cldice = self.cldice_loss(pred, target)
        boundary = self.boundary_loss(pred, target)
        tversky = self.tversky_loss(pred, target)
        ss = self.ss_loss(pred, target)
        
        total_loss = (self.dice_weight * dice +
                     self.cldice_weight * cldice +
                     self.boundary_weight * boundary +
                     self.tversky_weight * tversky +
                     self.ss_weight * ss)
        
        return total_loss

class LightweightVesselLoss(nn.Module):
    """轻量级血管损失 - 只使用最核心的损失组合"""
    
    def __init__(self, 
                 dice_weight=0.4,     # Dice损失
                 cldice_weight=0.5,   # clDice损失(主要)
                 boundary_weight=0.1, # 少量边界损失
                 smooth=1e-6):
        super(LightweightVesselLoss, self).__init__()
        
        self.dice_loss = DiceLoss(smooth=smooth)
        self.cldice_loss = clDiceLoss(smooth=smooth)
        self.boundary_loss = BoundaryLoss(smooth=smooth)
        
        self.dice_weight = dice_weight
        self.cldice_weight = cldice_weight
        self.boundary_weight = boundary_weight
        
        print(f"🩸 轻量级血管损失函数:")
        print(f"   • Dice:     {self.dice_weight:.1f}")
        print(f"   • clDice:   {self.cldice_weight:.1f} (主要)")
        print(f"   • Boundary: {self.boundary_weight:.1f}")
        
    def forward(self, pred, target):
        dice = self.dice_loss(pred, target)
        cldice = self.cldice_loss(pred, target)
        boundary = self.boundary_loss(pred, target)
        
        total_loss = (self.dice_weight * dice +
                     self.cldice_weight * cldice +
                     self.boundary_weight * boundary)
        
        return total_loss

class OptimizedVesselLoss(nn.Module):
    """优化版血管损失 - 添加Hausdorff损失并调优权重配置"""
    
    def __init__(self, 
                 dice_weight=0.2,       # Dice损失权重 (降低)
                 cldice_weight=0.4,     # clDice损失权重 (提升，最重要)
                 boundary_weight=0.2,   # 边界损失权重 (提升)
                 tversky_weight=0.1,    # Tversky损失权重 (降低)
                 hausdorff_weight=0.05, # Hausdorff距离损失 (新增)
                 ss_weight=0.05,        # SS损失权重 (降低)
                 smooth=1e-6):
        super(OptimizedVesselLoss, self).__init__()
        
        # 原有损失函数
        self.dice_loss = DiceLoss(smooth=smooth)
        self.cldice_loss = clDiceLoss(smooth=smooth)
        self.boundary_loss = BoundaryLoss(smooth=smooth)
        self.tversky_loss = TverskyLoss(alpha=0.3, beta=0.7, smooth=smooth)
        self.ss_loss = SSLoss(r=0.1, smooth=smooth)
        
        # 新增Hausdorff距离损失
        self.hausdorff_loss = HausdorffLoss(alpha=2.0, smooth=smooth)
        
        # 权重配置
        self.dice_weight = dice_weight
        self.cldice_weight = cldice_weight
        self.boundary_weight = boundary_weight
        self.tversky_weight = tversky_weight
        self.hausdorff_weight = hausdorff_weight
        self.ss_weight = ss_weight
        
        # 验证权重总和
        total_weight = sum([dice_weight, cldice_weight, boundary_weight, 
                           tversky_weight, hausdorff_weight, ss_weight])
        
        print(f"🩸 优化版血管损失函数 (总权重: {total_weight:.2f}):")
        print(f"   • clDice:    {self.cldice_weight:.2f} (血管拓扑，最重要)")
        print(f"   • Dice:      {self.dice_weight:.2f} (整体重叠)")
        print(f"   • Boundary:  {self.boundary_weight:.2f} (边界精度)")
        print(f"   • Tversky:   {self.tversky_weight:.2f} (FP/FN平衡)")
        print(f"   • Hausdorff: {self.hausdorff_weight:.2f} (形状完整性，新增)")
        print(f"   • SS:        {self.ss_weight:.2f} (敏感性/特异性)")
        
        if abs(total_weight - 1.0) > 0.01:
            print(f"   ⚠️  警告: 权重总和为 {total_weight:.3f}，建议调整为1.0")
        
    def forward(self, pred, target):
        # 计算各个损失分量
        dice = self.dice_loss(pred, target)
        cldice = self.cldice_loss(pred, target)  # 使用内置的forward方法
        boundary = self.boundary_loss(pred, target)
        tversky = self.tversky_loss(pred, target)
        hausdorff = self.hausdorff_loss(pred, target)
        ss = self.ss_loss(pred, target)
        
        # 加权组合
        total_loss = (self.dice_weight * dice +
                     self.cldice_weight * cldice +
                     self.boundary_weight * boundary +
                     self.tversky_weight * tversky +
                     self.hausdorff_weight * hausdorff +
                     self.ss_weight * ss)
        
        return total_loss
    
    def get_loss_details(self, pred, target):
        """获取各损失分量的详细信息，用于调试和分析"""
        with torch.no_grad():
            dice = self.dice_loss(pred, target)
            cldice = self.cldice_loss(pred, target)
            boundary = self.boundary_loss(pred, target)
            tversky = self.tversky_loss(pred, target)
            hausdorff = self.hausdorff_loss(pred, target)
            ss = self.ss_loss(pred, target)
            
            total = self.forward(pred, target)
            
            return {
                'total': total.item(),
                'dice': dice.item(),
                'cldice': cldice.item(),
                'boundary': boundary.item(),
                'tversky': tversky.item(),
                'hausdorff': hausdorff.item(),
                'ss': ss.item(),
                'weighted': {
                    'dice': (self.dice_weight * dice).item(),
                    'cldice': (self.cldice_weight * cldice).item(),
                    'boundary': (self.boundary_weight * boundary).item(),
                    'tversky': (self.tversky_weight * tversky).item(),
                    'hausdorff': (self.hausdorff_weight * hausdorff).item(),
                    'ss': (self.ss_weight * ss).item(),
                }
            }

class AdaptiveVesselLoss(nn.Module):
    """自适应血管损失 - 根据训练阶段动态调整权重"""
    
    def __init__(self, 
                 base_dice_weight=0.2,
                 base_cldice_weight=0.4,
                 base_boundary_weight=0.2,
                 base_tversky_weight=0.1,
                 base_hausdorff_weight=0.05,
                 base_ss_weight=0.05,
                 smooth=1e-6):
        super(AdaptiveVesselLoss, self).__init__()
        
        # 基础损失函数
        self.dice_loss = DiceLoss(smooth=smooth)
        self.cldice_loss = clDiceLoss(smooth=smooth)
        self.boundary_loss = BoundaryLoss(smooth=smooth)
        self.tversky_loss = TverskyLoss(alpha=0.3, beta=0.7, smooth=smooth)
        self.hausdorff_loss = HausdorffLoss(alpha=2.0, smooth=smooth)
        self.ss_loss = SSLoss(r=0.1, smooth=smooth)
        
        # 基础权重
        self.base_weights = {
            'dice': base_dice_weight,
            'cldice': base_cldice_weight,
            'boundary': base_boundary_weight,
            'tversky': base_tversky_weight,
            'hausdorff': base_hausdorff_weight,
            'ss': base_ss_weight
        }
        
        print(f"🩸 自适应血管损失函数:")
        print(f"   • 训练初期: 重点关注基础分割")
        print(f"   • 训练中期: 平衡各项损失")
        print(f"   • 训练后期: 强化拓扑和边界")
        
    def forward(self, pred, target, epoch=None, total_epochs=100):
        # 计算基础损失
        dice = self.dice_loss(pred, target)
        cldice = self.cldice_loss(pred, target)
        boundary = self.boundary_loss(pred, target)
        tversky = self.tversky_loss(pred, target)
        hausdorff = self.hausdorff_loss(pred, target)
        ss = self.ss_loss(pred, target)
        
        # 动态权重调整
        if epoch is not None:
            progress = epoch / total_epochs
            weights = self._get_adaptive_weights(progress)
        else:
            weights = self.base_weights
        
        # 加权组合
        total_loss = (weights['dice'] * dice +
                     weights['cldice'] * cldice +
                     weights['boundary'] * boundary +
                     weights['tversky'] * tversky +
                     weights['hausdorff'] * hausdorff +
                     weights['ss'] * ss)
        
        return total_loss
    
    def _get_adaptive_weights(self, progress):
        """根据训练进度自适应调整权重"""
        if progress < 0.3:
            # 训练初期: 重点关注基础Dice和clDice
            return {
                'dice': 0.3,
                'cldice': 0.4,
                'boundary': 0.15,
                'tversky': 0.1,
                'hausdorff': 0.03,
                'ss': 0.02
            }
        elif progress < 0.7:
            # 训练中期: 使用基础权重
            return self.base_weights
        else:
            # 训练后期: 强化拓扑和边界精度
            return {
                'dice': 0.15,
                'cldice': 0.45,
                'boundary': 0.25,
                'tversky': 0.08,
                'hausdorff': 0.05,
                'ss': 0.02
            }

class ConnectivityLoss(nn.Module):
    """连通性损失 - 专门处理血管断裂问题"""
    
    def __init__(self, smooth=1e-6):
        super(ConnectivityLoss, self).__init__()
        self.smooth = smooth
    
    def get_skeleton_connectivity(self, x):
        """获取骨架连通性特征"""
        # 获取骨架
        skeleton = self.soft_skeletonize(x)
        
        # 计算连通分量
        # 使用morphological operations检测端点和交叉点
        
        # 结构元素
        kernel = torch.ones(3, 3, device=x.device, dtype=x.dtype).unsqueeze(0).unsqueeze(0)
        
        # 计算邻域点数
        neighbor_count = F.conv2d(skeleton, kernel, padding=1)
        
        # 端点：邻域中只有1个点（自己除外）
        endpoints = ((neighbor_count == 2) & (skeleton > 0.5)).float()
        
        # 交叉点：邻域中有3个或更多点
        crossings = ((neighbor_count >= 4) & (skeleton > 0.5)).float()
        
        return skeleton, endpoints, crossings
    
    def soft_skeletonize(self, x, thresh_width=5):
        """软骨架化操作"""
        for i in range(thresh_width):
            min_pool_x = F.max_pool2d(x * -1, (3, 3), 1, 1) * -1
            contour = F.relu(F.max_pool2d(min_pool_x, (3, 3), 1, 1) - min_pool_x)
            x = F.relu(x - contour)
        return x
    
    def forward(self, pred, target):
        # 获取预测和目标的骨架特征
        pred_skeleton, pred_endpoints, pred_crossings = self.get_skeleton_connectivity(pred)
        target_skeleton, target_endpoints, target_crossings = self.get_skeleton_connectivity(target)
        
        # 骨架匹配损失
        skeleton_loss = F.mse_loss(pred_skeleton, target_skeleton)
        
        # 端点保持损失
        endpoint_loss = F.mse_loss(pred_endpoints, target_endpoints)
        
        # 交叉点保持损失
        crossing_loss = F.mse_loss(pred_crossings, target_crossings)
        
        # 组合损失
        connectivity_loss = 0.6 * skeleton_loss + 0.2 * endpoint_loss + 0.2 * crossing_loss
        
        return connectivity_loss

class UltimateVesselLoss(nn.Module):
    """终极血管损失 - 针对OCTA数据的全面优化"""
    
    def __init__(self, 
                 dice_weight=0.15,         # Dice损失权重 (进一步降低)
                 cldice_weight=0.35,       # clDice损失权重 (保持重要)
                 boundary_weight=0.2,      # 边界损失权重 (保持)
                 tversky_weight=0.1,       # Tversky损失权重 (保持)
                 hausdorff_weight=0.05,    # Hausdorff距离损失
                 connectivity_weight=0.1,  # 连通性损失 (新增)
                 ss_weight=0.05,           # SS损失权重 (保持)
                 smooth=1e-6):
        super(UltimateVesselLoss, self).__init__()
        
        # 原有损失函数
        self.dice_loss = DiceLoss(smooth=smooth)
        self.cldice_loss = clDiceLoss(smooth=smooth)
        self.boundary_loss = BoundaryLoss(smooth=smooth)
        self.tversky_loss = TverskyLoss(alpha=0.3, beta=0.7, smooth=smooth)
        self.hausdorff_loss = HausdorffLoss(alpha=2.0, smooth=smooth)
        self.ss_loss = SSLoss(r=0.1, smooth=smooth)
        
        # 新增连通性损失
        self.connectivity_loss = ConnectivityLoss(smooth=smooth)
        
        # 权重配置
        self.dice_weight = dice_weight
        self.cldice_weight = cldice_weight
        self.boundary_weight = boundary_weight
        self.tversky_weight = tversky_weight
        self.hausdorff_weight = hausdorff_weight
        self.connectivity_weight = connectivity_weight
        self.ss_weight = ss_weight
        
        # 验证权重总和
        total_weight = sum([dice_weight, cldice_weight, boundary_weight, 
                           tversky_weight, hausdorff_weight, connectivity_weight, ss_weight])
        
        print(f"🩸 终极血管损失函数 (总权重: {total_weight:.2f}):")
        print(f"   • clDice:       {self.cldice_weight:.2f} (血管拓扑，最重要)")
        print(f"   • Boundary:     {self.boundary_weight:.2f} (边界精度)")
        print(f"   • Dice:         {self.dice_weight:.2f} (整体重叠)")
        print(f"   • Connectivity: {self.connectivity_weight:.2f} (连通性，新增)")
        print(f"   • Tversky:      {self.tversky_weight:.2f} (FP/FN平衡)")
        print(f"   • Hausdorff:    {self.hausdorff_weight:.2f} (形状完整性)")
        print(f"   • SS:           {self.ss_weight:.2f} (敏感性/特异性)")
        
        if abs(total_weight - 1.0) > 0.01:
            print(f"   ⚠️  警告: 权重总和为 {total_weight:.3f}，建议调整为1.0")
        
    def forward(self, pred, target):
        # 计算各个损失分量
        dice = self.dice_loss(pred, target)
        cldice = self.cldice_loss(pred, target)
        boundary = self.boundary_loss(pred, target)
        tversky = self.tversky_loss(pred, target)
        hausdorff = self.hausdorff_loss(pred, target)
        connectivity = self.connectivity_loss(pred, target)
        ss = self.ss_loss(pred, target)
        
        # 加权组合
        total_loss = (self.dice_weight * dice +
                     self.cldice_weight * cldice +
                     self.boundary_weight * boundary +
                     self.tversky_weight * tversky +
                     self.hausdorff_weight * hausdorff +
                     self.connectivity_weight * connectivity +
                     self.ss_weight * ss)
        
        return total_loss
    
    def get_loss_details(self, pred, target):
        """获取各损失分量的详细信息，用于调试和分析"""
        with torch.no_grad():
            dice = self.dice_loss(pred, target)
            cldice = self.cldice_loss(pred, target)
            boundary = self.boundary_loss(pred, target)
            tversky = self.tversky_loss(pred, target)
            hausdorff = self.hausdorff_loss(pred, target)
            connectivity = self.connectivity_loss(pred, target)
            ss = self.ss_loss(pred, target)
            
            total = self.forward(pred, target)
            
            return {
                'total': total.item(),
                'dice': dice.item(),
                'cldice': cldice.item(),
                'boundary': boundary.item(),
                'tversky': tversky.item(),
                'hausdorff': hausdorff.item(),
                'connectivity': connectivity.item(),
                'ss': ss.item(),
                'weighted': {
                    'dice': (self.dice_weight * dice).item(),
                    'cldice': (self.cldice_weight * cldice).item(),
                    'boundary': (self.boundary_weight * boundary).item(),
                    'tversky': (self.tversky_weight * tversky).item(),
                    'hausdorff': (self.hausdorff_weight * hausdorff).item(),
                    'connectivity': (self.connectivity_weight * connectivity).item(),
                    'ss': (self.ss_weight * ss).item(),
                }
            }

class DynamicLoss(nn.Module):
    """动态损失函数 - 根据训练阶段动态调整损失组件权重
    
    支持两种模式:
    1. progressive_simplify: 从复杂到简单 (全部组件 -> 逐渐减少 -> 最后只有dice)
    2. progressive_enhance: 从简单到复杂 (开始只有dice -> 逐渐增加复杂组件)
    """
    
    def __init__(self, 
                 mode='progressive_simplify',  # 'progressive_simplify' 或 'progressive_enhance'
                 smooth=1e-6):
        super(DynamicLoss, self).__init__()
        self.smooth = smooth
        self.mode = mode
        
        # 初始化各损失函数组件
        self.dice_loss = DiceLoss(smooth=smooth)
        self.cldice_loss = clDiceLoss(smooth=smooth)
        self.boundary_loss = BoundaryLoss(smooth=smooth)
        self.tversky_loss = TverskyLoss(smooth=smooth)
        self.hausdorff_loss = HausdorffLoss(smooth=smooth)
        self.connectivity_loss = ConnectivityLoss(smooth=smooth)
        self.focal_loss = FocalDiceLoss(smooth=smooth)
        self.ss_loss = SSLoss(smooth=smooth)
        
        # 记录当前epoch和总epochs
        self.current_epoch = 0
        self.total_epochs = 100
        
        print(f"🔄 创建动态损失函数，模式: {mode}")
    
    def update_epoch(self, current, total=None):
        """更新当前训练阶段
        
        Args:
            current: 当前epoch
            total: 总epochs数
        """
        self.current_epoch = current
        if total is not None:
            self.total_epochs = total
    
    def get_progress(self):
        """获取训练进度 (0~1之间)"""
        return min(1.0, max(0.0, self.current_epoch / self.total_epochs))
    
    def get_component_weights(self):
        """根据训练进度和模式获取各组件权重"""
        progress = self.get_progress()
        
        if self.mode == 'progressive_simplify':
            # 从复杂到简单: 开始时使用所有损失函数，逐渐过渡到只有dice
            dice_weight = 0.2 + 0.8 * progress  # 0.2 -> 1.0
            cldice_weight = 0.4 * (1 - progress)  # 0.4 -> 0.0
            boundary_weight = 0.2 * (1 - progress)  # 0.2 -> 0.0
            tversky_weight = 0.1 * (1 - progress)  # 0.1 -> 0.0
            hausdorff_weight = 0.05 * (1 - progress)  # 0.05 -> 0.0
            connectivity_weight = 0.0  # 不使用连通性损失
            focal_weight = 0.05 * (1 - progress)  # 0.05 -> 0.0
            ss_weight = 0.0  # 不使用SS损失
        
        elif self.mode == 'progressive_enhance':
            # 从简单到复杂: 开始时只使用dice，逐渐添加更多损失函数
            dice_weight = 1.0 - 0.8 * progress  # 1.0 -> 0.2
            cldice_weight = 0.4 * progress  # 0.0 -> 0.4
            boundary_weight = 0.2 * progress  # 0.0 -> 0.2
            tversky_weight = 0.1 * progress  # 0.0 -> 0.1
            hausdorff_weight = 0.05 * progress  # 0.0 -> 0.05
            connectivity_weight = 0.0  # 不使用连通性损失
            focal_weight = 0.05 * progress  # 0.0 -> 0.05
            ss_weight = 0.0  # 不使用SS损失
        
        else:  # 默认为平衡模式
            # 使用固定权重
            dice_weight = 0.2
            cldice_weight = 0.4
            boundary_weight = 0.2
            tversky_weight = 0.1
            hausdorff_weight = 0.05
            connectivity_weight = 0.0
            focal_weight = 0.05
            ss_weight = 0.0
        
        return {
            'dice': dice_weight,
            'cldice': cldice_weight,
            'boundary': boundary_weight,
            'tversky': tversky_weight,
            'hausdorff': hausdorff_weight,
            'connectivity': connectivity_weight,
            'focal': focal_weight,
            'ss': ss_weight
        }
    
    def forward(self, pred, target):
        # 获取当前阶段的损失权重
        weights = self.get_component_weights()
        
        # 计算各损失分量
        losses = {}
        if weights['dice'] > 0:
            losses['dice'] = self.dice_loss(pred, target) * weights['dice']
        
        if weights['cldice'] > 0:
            losses['cldice'] = self.cldice_loss(pred, target) * weights['cldice']
        
        if weights['boundary'] > 0:
            losses['boundary'] = self.boundary_loss(pred, target) * weights['boundary']
        
        if weights['tversky'] > 0:
            losses['tversky'] = self.tversky_loss(pred, target) * weights['tversky']
        
        if weights['hausdorff'] > 0:
            losses['hausdorff'] = self.hausdorff_loss(pred, target) * weights['hausdorff']
        
        if weights['connectivity'] > 0:
            losses['connectivity'] = self.connectivity_loss(pred, target) * weights['connectivity']
        
        if weights['focal'] > 0:
            losses['focal'] = self.focal_loss(pred, target) * weights['focal']
        
        if weights['ss'] > 0:
            losses['ss'] = self.ss_loss(pred, target) * weights['ss']
        
        # 合并所有损失
        total_loss = sum(losses.values())
        
        return total_loss
    
    def get_loss_details(self, pred, target):
        """获取各损失组件的详细信息"""
        weights = self.get_component_weights()
        details = {}
        
        # 计算各损失分量的原始值和加权值
        if weights['dice'] > 0:
            dice_value = self.dice_loss(pred, target)
            details['dice'] = dice_value.item()
            details['weighted_dice'] = (dice_value * weights['dice']).item()
        
        if weights['cldice'] > 0:
            cldice_value = self.cldice_loss(pred, target)
            details['cldice'] = cldice_value.item()
            details['weighted_cldice'] = (cldice_value * weights['cldice']).item()
        
        if weights['boundary'] > 0:
            boundary_value = self.boundary_loss(pred, target)
            details['boundary'] = boundary_value.item()
            details['weighted_boundary'] = (boundary_value * weights['boundary']).item()
        
        if weights['tversky'] > 0:
            tversky_value = self.tversky_loss(pred, target)
            details['tversky'] = tversky_value.item()
            details['weighted_tversky'] = (tversky_value * weights['tversky']).item()
        
        if weights['hausdorff'] > 0:
            hausdorff_value = self.hausdorff_loss(pred, target)
            details['hausdorff'] = hausdorff_value.item()
            details['weighted_hausdorff'] = (hausdorff_value * weights['hausdorff']).item()
        
        if weights['connectivity'] > 0:
            connectivity_value = self.connectivity_loss(pred, target)
            details['connectivity'] = connectivity_value.item()
            details['weighted_connectivity'] = (connectivity_value * weights['connectivity']).item()
        
        if weights['focal'] > 0:
            focal_value = self.focal_loss(pred, target)
            details['focal'] = focal_value.item()
            details['weighted_focal'] = (focal_value * weights['focal']).item()
        
        if weights['ss'] > 0:
            ss_value = self.ss_loss(pred, target)
            details['ss'] = ss_value.item()
            details['weighted_ss'] = (ss_value * weights['ss']).item()
        
        # 添加当前进度和模式信息
        details['progress'] = self.get_progress()
        details['mode'] = self.mode
        
        # 合并所有损失
        total_loss = sum([v for k, v in details.items() if k.startswith('weighted_')])
        details['total_loss'] = total_loss
        
        return details

class FixedCombinedLoss(nn.Module):
    """固定组合损失函数 - 0.8 Dice + 0.1 clDice + 0.1 Lovasz"""
    def __init__(self, smooth=1e-6):
        super(FixedCombinedLoss, self).__init__()
        self.dice_loss = DiceLoss(smooth=smooth)
        self.cldice_loss = clDiceLoss(smooth=smooth)
        self.Lovasz_loss = LovaszSoftmaxLoss(smooth=smooth)
        
    def forward(self, pred, target):
        dice = self.dice_loss(pred, target)
        cldice = self.cldice_loss(pred, target)
        Lovasz = self.Lovasz_loss(pred, target)
        
        total_loss = 0.8 * dice + 0.1 * cldice + 0.1 * Lovasz
        return total_loss

# =================== 便捷调用函数 ===================

def get_loss_function(loss_name, **kwargs):
    """便捷的损失函数获取接口
    
    Args:
        loss_name (str): 损失函数名称
        **kwargs: 损失函数的参数
        
    Returns:
        损失函数实例
    """
    loss_dict = {
        # 基础损失
        'dice': DiceLoss,
        'cldice': clDiceLoss,
        'focal': FocalLoss,
        'ce': CrossEntropyLoss,
        'weighted_ce': WeightedCrossEntropyLoss,
        
        # 高级损失
        'tversky': TverskyLoss,
        'focal_tversky': FocalTverskyLoss,
        'iou': IoULoss,
        'ss': SSLoss,
        'unified_focal': UnifiedFocalLoss,
        'combo': ComboLoss,
        'lovasz': LovaszSoftmaxLoss,
        'hausdorff': HausdorffLoss,
        'asymmetric': AsymmetricLoss,
        
        # 边界增强损失
        'boundary': BoundaryLoss,
        'focal_dice': FocalDiceLoss,
        
        # 组合损失
        'conservative': ConservativeEnhancedLoss,
        'boundary_enhanced': EnhancedBoundaryLoss,
        'minimal': MinimalEnhancedLoss,
        'ultra_conservative': UltraConservativeLoss,
        
        # 🩸 血管分割专用损失
        'vessel': VesselSegmentationLoss,
        'vessel_enhanced': EnhancedVesselLoss,
        'vessel_lightweight': LightweightVesselLoss,
        'optimized': OptimizedVesselLoss,
        'adaptive': AdaptiveVesselLoss,
        'connectivity': ConnectivityLoss,
        'ultimate': UltimateVesselLoss,
        'dynamic': DynamicLoss,
        'fixed_combined': FixedCombinedLoss,
    }
    
    if loss_name not in loss_dict:
        available = ', '.join(loss_dict.keys())
        raise ValueError(f"Loss '{loss_name}' not found. Available: {available}")
    
    return loss_dict[loss_name](**kwargs)

def create_medical_loss_combination(loss_types, weights=None):
    """创建医学图像分割的组合损失函数
    
    Args:
        loss_types (list): 损失函数名称列表
        weights (list): 对应的权重列表
        
    Returns:
        组合损失函数
    """
    if weights is None:
        weights = [1.0 / len(loss_types)] * len(loss_types)
    
    if len(loss_types) != len(weights):
        raise ValueError("损失函数数量和权重数量不匹配")
    
    class CombinedLoss(nn.Module):
        def __init__(self):
            super(CombinedLoss, self).__init__()
            self.losses = nn.ModuleList([get_loss_function(name) for name in loss_types])
            self.weights = weights
            
        def forward(self, pred, target):
            total_loss = 0
            for loss_fn, weight in zip(self.losses, self.weights):
                total_loss += weight * loss_fn(pred, target)
            return total_loss
    
    return CombinedLoss()

# =================== 使用示例 ===================
"""
使用示例：

# 1. 单一损失函数
dice_loss = get_loss_function('dice')
focal_loss = get_loss_function('focal', alpha=0.25, gamma=2.0)
tversky_loss = get_loss_function('tversky', alpha=0.3, beta=0.7)

# 2. 组合损失函数
combo_loss = create_medical_loss_combination(
    loss_types=['dice', 'focal', 'boundary'],
    weights=[0.5, 0.3, 0.2]
)

# 3. 医学分割常用组合
octa_loss = create_medical_loss_combination(
    loss_types=['dice', 'cldice', 'focal_tversky'],
    weights=[0.4, 0.4, 0.2]
)

# 4. 血管分割专用
vessel_loss = create_medical_loss_combination(
    loss_types=['dice', 'cldice', 'boundary', 'ss'],
    weights=[0.3, 0.3, 0.2, 0.2]
)
"""

# =================== 测试代码 ===================
if __name__ == "__main__":
    print("🧪 测试医学图像分割损失函数")
    print("="*50)
    
    # 创建测试数据
    batch_size, height, width = 2, 64, 64
    pred = torch.sigmoid(torch.randn(batch_size, 1, height, width))
    target = torch.randint(0, 2, (batch_size, 1, height, width)).float()
    
    print(f"测试数据形状: pred={pred.shape}, target={target.shape}")
    print()
    
    # 测试单一损失函数
    print("📊 单一损失函数测试:")
    single_losses = [
        'dice', 'ce', 'focal', 'tversky', 'focal_tversky', 
        'iou', 'ss', 'boundary', 'asymmetric'
    ]
    
    for loss_name in single_losses:
        try:
            loss_fn = get_loss_function(loss_name)
            loss_value = loss_fn(pred, target)
            print(f"  {loss_name:15s}: {loss_value.item():.4f}")
        except Exception as e:
            print(f"  {loss_name:15s}: Error - {e}")
    
    print()
    
    # 测试组合损失函数
    print("🔗 组合损失函数测试:")
    
    # OCTA血管分割专用组合
    octa_loss = create_medical_loss_combination(
        loss_types=['dice', 'cldice', 'boundary'],
        weights=[0.5, 0.3, 0.2]
    )
    octa_value = octa_loss(pred, target)
    print(f"  OCTA血管分割组合: {octa_value.item():.4f}")
    
    # 一般医学分割组合
    general_loss = create_medical_loss_combination(
        loss_types=['dice', 'focal', 'iou'],
        weights=[0.4, 0.4, 0.2]
    )
    general_value = general_loss(pred, target)
    print(f"  一般医学分割组合: {general_value.item():.4f}")
    
    # 边界敏感组合
    boundary_loss = create_medical_loss_combination(
        loss_types=['dice', 'boundary', 'hausdorff'],
        weights=[0.5, 0.3, 0.2]
    )
    boundary_value = boundary_loss(pred, target)
    print(f"  边界敏感组合    : {boundary_value.item():.4f}")
    
    print()
    print("✅ 所有损失函数测试完成！")
    print()
    print("💡 使用建议:")
    print("  • 血管分割: dice + cldice + boundary")
    print("  • 小目标分割: focal + tversky + iou") 
    print("  • 边界敏感任务: dice + boundary + hausdorff")
    print("  • 类别不平衡: weighted_ce + focal + asymmetric")
    print("="*50)
