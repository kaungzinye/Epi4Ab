import torch
from torch import nn
import torch.nn.functional as F
from typing import List, Optional

class CustomMSELoss(nn.MSELoss):
    def __init__(self):
        super().__init__()
    
    def forward(self, input: torch.tensor, target: torch.Tensor) -> torch.tensor:
        return F.mse_loss(input.reshape(-1), target.float())
    
class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Where:
    - p_t is the predicted probability for the true class
    - alpha_t is the weighting factor for class t
    - gamma is the focusing parameter (gamma >= 0)
    
    Higher gamma focuses more on hard examples.
    """
    def __init__(self, alpha: Optional[List[float]] = None, gamma: float = 2.0, device=None):
        super().__init__()
        self.gamma = gamma
        self.device = device
        
        if alpha is None:
            # Default: equal weights
            self.alpha = torch.tensor([1.0, 1.0, 1.0]).to(device) if device else torch.tensor([1.0, 1.0, 1.0])
        else:
            self.alpha = torch.tensor(alpha).to(device) if device else torch.tensor(alpha)
    
    def forward(self, logits, targets):
        """
        Compute focal loss.
        
        Args:
            logits: Raw model predictions (batch_size, num_classes)
            targets: Ground truth class indices (batch_size)
        
        Returns:
            Focal loss value
        """
        # Convert logits to probabilities
        probs = torch.softmax(logits, dim=-1)
        
        # Get probability of true class for each sample
        batch_size = targets.size(0)
        class_indices = targets.long()
        p_t = probs.gather(1, class_indices.unsqueeze(1)).squeeze(1)
        
        # Get alpha for each sample
        alpha_t = self.alpha[class_indices]
        
        # Compute focal loss: -alpha_t * (1 - p_t)^gamma * log(p_t)
        # Add small epsilon to log for numerical stability
        log_p_t = torch.log(p_t + 1e-8)
        focal_weight = (1 - p_t) ** self.gamma
        focal_loss = -alpha_t * focal_weight * log_p_t
        
        return focal_loss.mean()


class HierarchicalCELoss(nn.Module):
    def __init__(self, cross_entropy_weight, device):
        super().__init__()
        self.reachability_matrix = torch.tensor([[1,1,1],
                                                 [0,1,0],
                                                 [0,0,1]]).float().to(device)
        '''
        [[1,1,1,1],
         [0,1,0,0],
         [0,0,1,0],
         [0,1,1,1]]
        '''
        self.cross_entropy_weight = torch.tensor(cross_entropy_weight).to(device)
        self.device = device
        
    def forward(self, logits, targets):
        """
        Hierarchical Cross-Entropy loss
        
        Args:
            logits: Raw model predictions (batch_size, num_classes)
            targets: Ground truth class indices (batch_size)
            reachability_matrix: Matrix encoding hierarchical relationships (num_classes, num_classes)
            weight: Optional class weights
        
        Returns:
            Hierarchical Cross-Entropy loss value
        https://github.com/microsoft/hce-classification
        """
        # Convert logits to probabilities using softmax
        cell_type_probs = torch.softmax(logits, dim=-1)
        
        # Propagate probabilities through the hierarchy using the reachability matrix
        cell_type_probs = torch.matmul(cell_type_probs, self.reachability_matrix.T)
        
        # Apply log transform (with numerical stability term) for NLL loss calculation
        cell_type_probs = torch.log(
            cell_type_probs + torch.tensor(1e-6, device=self.device)
        )
        
        # Calculate negative log-likelihood loss with optional class weights
        hce_loss = F.nll_loss(cell_type_probs, targets, weight=self.cross_entropy_weight).to(self.device)
        return hce_loss
    
def get_loss_function(loss_function, 
                      cross_entropy_weight,
                      device,
                      focal_alpha=None,
                      focal_gamma=2.0):
    if loss_function == 'cross_entropy': 
        if cross_entropy_weight:
            return nn.CrossEntropyLoss(weight=torch.tensor(cross_entropy_weight).to(device))
        else:
            return nn.CrossEntropyLoss()
    elif loss_function == 'mse':
        return CustomMSELoss()
    elif loss_function == 'hce':
        return HierarchicalCELoss(cross_entropy_weight, device)
    elif loss_function == 'focal':
        # Use focal_alpha if provided, otherwise use cross_entropy_weight as alpha
        alpha = focal_alpha if focal_alpha is not None else cross_entropy_weight
        return FocalLoss(alpha=alpha, gamma=focal_gamma, device=device)
    
