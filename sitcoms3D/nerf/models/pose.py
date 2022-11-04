import torch
import torch.nn as nn
# from kornia import angle_axis_to_rotation_matrix
from kornia.geometry.conversions import angle_axis_to_rotation_matrix
import math


class LearnFocal(nn.Module):
    def __init__(self, num_cams, learn_f):
        super(LearnFocal, self).__init__()
        self.fx = nn.Parameter(torch.ones(size=(num_cams, 1), dtype=torch.float32), requires_grad=learn_f)  # (1, )
        self.ones = nn.Parameter(torch.ones(size=(num_cams, 1), dtype=torch.float32), requires_grad=False)

    def forward(self):  # the i=None is just to enable multi-gpu training
        return torch.cat([self.fx, self.fx, self.ones], 1)


class LearnRot(nn.Module):
    def __init__(self, num_cams, learn_r):
        super(LearnRot, self).__init__()
        tensor = torch.zeros(size=(num_cams, 3), dtype=torch.float32)
        # NOTE(ethan): why don't they do this in the nermm code?
        tensor[:, 2] = 2*math.pi  # set the rotations to be the identity
        self.rot = nn.Parameter(tensor, requires_grad=learn_r)  # (1, )

    def forward(self):  # the i=None is just to enable multi-gpu training
        return self.rot


class LearnTrans(nn.Module):
    def __init__(self, num_cams, learn_t):
        super(LearnTrans, self).__init__()
        self.trans = nn.Parameter(torch.zeros(size=(num_cams, 3), dtype=torch.float32), requires_grad=learn_t)  # (N, 3)

    def forward(self):
        return self.trans


class LearnPose(nn.Module):
    """https://github.com/ActiveVisionLab/nerfmm/blob/27faab66a927ea14259125e1140231f0c8f6d14c/models/poses.py
    """

    def __init__(self, num_cams, learn_r, learn_t, init_c2w=None):
        """
        :param num_cams:
        :param learn_R:  True/False
        :param learn_t:  True/False
        :param init_c2w: (N, 4, 4) torch tensor
        """
        super(LearnPose, self).__init__()
        self.num_cams = num_cams
        self.init_c2w = None
        if init_c2w is not None:
            self.init_c2w = nn.Parameter(init_c2w, requires_grad=False)

        self.r = LearnRot(self.num_cams, learn_r)
        self.t = LearnTrans(self.num_cams, learn_t)

    @staticmethod
    def make_c2w(r, t):
        R = angle_axis_to_rotation_matrix(r)  # (N, 3, 3)
        c2w = torch.cat([R, t.unsqueeze(2)], dim=2)  # (N, 3, 4)
        return c2w

    def forward(self):
        """Returns a (N, 3, 4) torch tensor.
        These are the current poses.
        """
        # assert self.init_c2w is not None, "init_c2w shouldn't be None!"

        r = self.r()  # (N, 3, ) axis-angle
        t = self.t()  # (N, 3, )
        c2w_delta = LearnPose.make_c2w(r, t)  # (N, 3, 4)
        # c2w = make_c2w(r, t)[:, :3]

        # learn a delta pose between init pose and target pose
        c2w = torch.bmm(c2w_delta, self.init_c2w)
        return c2w, c2w_delta


if __name__ == "__main__":
    # identity rotation as angle_axis
    import kornia
    R = torch.eye(3).unsqueeze(0)
    angle_axis = kornia.rotation_matrix_to_angle_axis(R)
    print(angle_axis)
    # results is [0, 0, pi]
