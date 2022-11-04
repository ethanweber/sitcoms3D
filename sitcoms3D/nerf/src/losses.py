import torch
from torch import nn

class ColorLoss(nn.Module):
    def __init__(self, coef=1):
        super().__init__()
        self.coef = coef
        self.loss = nn.MSELoss(reduction='mean')

    def forward(self, inputs, targets):
        loss = self.loss(inputs['rgb_coarse'], targets)
        if 'rgb_fine' in inputs:
            loss += self.loss(inputs['rgb_fine'], targets)

        return self.coef * loss


class NerfWLoss(nn.Module):
    """
    Equation 13 in the NeRF-W paper.
    Name abbreviations:
        c_l: coarse color loss
        f_l: fine color loss (1st term in equation 13)
        b_l: beta loss (2nd term in equation 13)
        s_l: sigma loss (3rd term in equation 13)
    """
    def __init__(self, coef=1, lambda_u=0.01):
        """
        lambda_u: in equation 13
        """
        super().__init__()
        self.coef = coef
        self.lambda_u = lambda_u

    def forward(self, inputs, targets, ray_mask):
        ray_mask_sum = ray_mask.sum() + 1e-20
        # if ray_mask_sum < len(inputs['rgb_fine']):
        #     print(ray_mask_sum)

        # print(inputs["transient_accumulation"].shape)
        

        ret = {}
        ret['c_l'] = 0.5 * (((inputs['rgb_coarse']-targets)**2) * ray_mask[:, None]).sum() / ray_mask_sum
        if 'rgb_fine' in inputs:
            if 'beta' not in inputs: # no transient head, normal MSE loss
                ret['f_l'] = 0.5 * (((inputs['rgb_fine']-targets)**2) * ray_mask[:, None]).sum() / ray_mask_sum
            else:
                ret['f_l'] = \
                    (((inputs['rgb_fine']-targets)**2/(2*inputs['beta'].unsqueeze(1)**2)) * ray_mask[:, None]).sum() / ray_mask_sum
                ret['b_l'] = 3 + (torch.log(inputs['beta']) * ray_mask).sum() / ray_mask_sum
                ret['s_l'] = self.lambda_u * inputs['transient_sigmas'].mean()

        for k, v in ret.items():
            ret[k] = self.coef * v

        return ret

loss_dict = {'color': ColorLoss,
             'nerfw': NerfWLoss}