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
    
    def __init__(self, alpha=0.25, gamma=2.0, smooth=1e-6):
        super(FocalDiceLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
    
    def forward(self, pred, target):
        intersection = (pred * target).sum()
        denominator = pred.sum() + target.sum()
        dice_score = (2. * intersection + self.smooth) / (denominator + self.smooth)
        dice_loss = 1. - dice_score
        
        focal_weight = self.alpha * (dice_loss ** self.gamma)
        
        return focal_weight * dice_loss

class BoundaryLoss(torch.nn.Module):
    
    def __init__(self, smooth=1e-6):
        super(BoundaryLoss, self).__init__()
        self.smooth = smooth
    
    def get_boundary(self, x):
        kernel_size = 3
        
        dilated = F.max_pool2d(x, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        
        eroded = -F.max_pool2d(-x, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        
        boundary = torch.abs(dilated - eroded)
        
        return boundary
    
    def forward(self, pred, target):
        pred_boundary = self.get_boundary(pred)
        target_boundary = self.get_boundary(target)
        
        boundary_loss = F.mse_loss(pred_boundary, target_boundary)
        
        return boundary_loss

class ConservativeEnhancedLoss(torch.nn.Module):
    
    def __init__(self, smooth=1e-6):
        super(ConservativeEnhancedLoss, self).__init__()
        self.smooth = smooth
        
        self.dice_loss = DiceLoss(smooth=smooth)
        self.cldice_loss = clDiceLoss(smooth=smooth)
        self.focal_dice_loss = FocalDiceLoss(smooth=smooth)
        
        self.dice_weight = 0.76     # 0.8 - 0.04
        self.cldice_weight = 0.19   # 0.2 - 0.01  
        self.focal_weight = 0.05    
        
    
    def forward(self, pred, target):
        dice = self.dice_loss(pred, target)
        cldice = self.cldice_loss.soft_cldice_loss(pred, target)
        focal = self.focal_dice_loss(pred, target)
        

        total_loss = (self.dice_weight * dice + 
                      self.cldice_weight * cldice + 
                      self.focal_weight * focal)
        
        return total_loss

class EnhancedBoundaryLoss(torch.nn.Module):
    
    def __init__(self, smooth=1e-6):
        super(EnhancedBoundaryLoss, self).__init__()
        self.smooth = smooth
        
        self.dice_loss = DiceLoss(smooth=smooth)
        self.cldice_loss = clDiceLoss(smooth=smooth)
        self.boundary_loss = BoundaryLoss(smooth=smooth)
        
        self.dice_weight = 0.72     
        self.cldice_weight = 0.18  
        self.boundary_weight = 0.10  
        
    
    def forward(self, pred, target):
        dice = self.dice_loss(pred, target)
        cldice = self.cldice_loss.soft_cldice_loss(pred, target)
        boundary = self.boundary_loss(pred, target)

        total_loss = (self.dice_weight * dice + 
                      self.cldice_weight * cldice + 
                      self.boundary_weight * boundary)
        
        return total_loss

class MinimalEnhancedLoss(nn.Module):
    def __init__(self):
        super(MinimalEnhancedLoss, self).__init__()
        self.dice_loss = DiceLoss()
        self.cldice_loss = clDiceLoss()
        self.focal_loss = FocalDiceLoss()
        
        self.dice_weight = 0.79   
        self.cldice_weight = 0.20  
        self.focal_weight = 0.01   
        
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
        
        self.dice_weight = 0.796   
        self.cldice_weight = 0.199
        self.focal_weight = 0.005 
        
    def forward(self, pred, target):
        dice = self.dice_loss(pred, target)
        cldice = self.cldice_loss(pred, target)  
        focal = self.focal_loss(pred, target)
        
        total_loss = (self.dice_weight * dice + 
                     self.cldice_weight * cldice + 
                     self.focal_weight * focal)
        
        return total_loss


class CrossEntropyLoss(nn.Module):
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
    def __init__(self, class_weights=None, reduction='mean'):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.class_weights = class_weights
        self.reduction = reduction
        
    def forward(self, pred, target):
        if self.class_weights is None:
            unique, counts = torch.unique(target, return_counts=True)
            total = target.numel()
            weights = total / (len(unique) * counts.float())
            self.class_weights = weights.to(pred.device)
        
        return F.cross_entropy(pred, target.long(), 
                             weight=self.class_weights,
                             reduction=self.reduction)

class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, smooth=1e-6):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha  
        self.beta = beta    
        self.smooth = smooth
        
    def forward(self, pred, target):
        TP = (pred * target).sum()
        FP = (pred * (1 - target)).sum()
        FN = ((1 - pred) * target).sum()
        
        tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        
        return 1 - tversky

class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, gamma=2.0, smooth=1e-6):
        super(FocalTverskyLoss, self).__init__()
        self.tversky_loss = TverskyLoss(alpha, beta, smooth)
        self.gamma = gamma
        
    def forward(self, pred, target):
        tversky_loss = self.tversky_loss(pred, target)
        focal_tversky = torch.pow(tversky_loss, self.gamma)
        
        return focal_tversky

class IoULoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(IoULoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, pred, target):
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum() - intersection
        
        iou = (intersection + self.smooth) / (union + self.smooth)
        
        return 1 - iou

class SSLoss(nn.Module):
    def __init__(self, r=0.1, smooth=1e-6):
        super(SSLoss, self).__init__()
        self.r = r  
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
        
        ss_loss = self.r * (1 - sensitivity) + (1 - self.r) * (1 - specificity)
        
        return ss_loss

class UnifiedFocalLoss(nn.Module):
    def __init__(self, weight=1.0, delta=0.6, gamma=2.0):
        super(UnifiedFocalLoss, self).__init__()
        self.weight = weight
        self.delta = delta
        self.gamma = gamma
        
    def forward(self, pred, target):

        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        
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
    def __init__(self, alpha=0.5, ce_ratio=0.5, smooth=1e-6):
        super(ComboLoss, self).__init__()
        self.alpha = alpha      
        self.ce_ratio = ce_ratio
        self.smooth = smooth
        
    def forward(self, pred, target):
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        dice_loss = 1 - dice
        
        bce = F.binary_cross_entropy_with_logits(pred, target)
    
        combo = (self.ce_ratio * bce) + ((1 - self.ce_ratio) * dice_loss)
        
        return combo

class LovaszSoftmaxLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(LovaszSoftmaxLoss, self).__init__()
        self.smooth = smooth
        
    def lovasz_grad(self, gt_sorted):
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
        
        grad = self.lovasz_grad(gt_sorted)
        loss = torch.dot(F.relu(errors_sorted), grad)
        
        return loss

class HausdorffLoss(nn.Module):
    def __init__(self, alpha=2.0, smooth=1e-6):
        super(HausdorffLoss, self).__init__()
        self.alpha = alpha
        self.smooth = smooth
        
    def compute_dtm(self, img_gt, out_shape):
        import scipy.ndimage as ndi
        
        fg_dtm = ndi.distance_transform_edt(img_gt)
        bg_dtm = ndi.distance_transform_edt(1 - img_gt)
        
        dtm = fg_dtm + bg_dtm
        return torch.tensor(dtm, dtype=torch.float32)
        
    def forward(self, pred, target):
        pred_edge = self.get_edge(pred)
        target_edge = self.get_edge(target)
        
        hausdorff_loss = F.mse_loss(pred_edge, target_edge)
        
        return hausdorff_loss
        
    def get_edge(self, x):
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
        
        edge_x = F.conv2d(x, sobel_x, padding=1)
        edge_y = F.conv2d(x, sobel_y, padding=1)
        
        edge = torch.sqrt(edge_x**2 + edge_y**2 + 1e-6)
        
        return edge

class AsymmetricLoss(nn.Module):
    def __init__(self, beta_pos=1.0, beta_neg=4.0, gamma_pos=2.0, gamma_neg=1.0):
        super(AsymmetricLoss, self).__init__()
        self.beta_pos = beta_pos
        self.beta_neg = beta_neg  
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        
    def forward(self, pred, target):
        prob = torch.sigmoid(pred)
        
        pos_loss = target * torch.pow(1 - prob, self.gamma_pos) * F.logsigmoid(pred)
        
        neg_loss = (1 - target) * torch.pow(prob, self.gamma_neg) * F.logsigmoid(-pred)
        
        loss = -(self.beta_pos * pos_loss + self.beta_neg * neg_loss)
        
        return loss.mean()





