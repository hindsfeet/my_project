import torch

class RBM():
  def __init__(self, nv, nh):
    """
    Function to initialize the constructor of the RBM - Restricted Boltzmann Machine
      (self.W - weights for the visible and hidden), self.a - hidden nodes, self.b - visible nods)

    Parameters
    ----------
      nv (torch): visible nodes
      nh (torch): hidden nodes

    References
    ----------
      __init__() - https://www.udemy.com/course/deeplearning/learn/lecture/6895676#overview
    """
    self.W = torch.randn(nh, nv)
    self.a = torch.randn(1, nh)
    self.b = torch.randn(1, nv)

  def sample_h(self, x: torch) -> torch:
    """
      Function to return the probability of the sample hidden nodes given the visible nodes

      Parameters
      ----------
        x (torch): the visible nodes

      Return
      ______
        p_h_given_v (torch),  p_h_given_v (torch): probability of hidden nodes given the visible data,
          both returning the same type, only the second one is using a torch.bernouli function

      References
      ----------
        sample_h() - https://www.udemy.com/course/deeplearning/learn/lecture/6895678#overview
      """
    wx = torch.mm(x, self.W.t())
    activation = wx + self.a.expand_as(wx)
    p_h_given_v = torch.sigmoid(activation)
    return p_h_given_v, torch.bernoulli(p_h_given_v)

  def sample_v(self, y: torch) -> torch:
    """
    Function to return the probability of the sample visible nodes given the hidden nodes

    Parameters
    ----------
      y (torch): the hidden nodes

    Return
    ______
      p_v_given_h (torch),  p_v_given_h (torch): probability of visible nodes given the hidden data,
        both returning the same type, only the second one is using a torch.bernouli function

    References
    ----------
      sample_v() - https://www.udemy.com/course/deeplearning/learn/lecture/6895682#overview
    """
    wy = torch.mm(y, self.W)
    activation = wy + self.b.expand_as(wy)
    p_v_given_h = torch.sigmoid(activation)
    return p_v_given_h, torch.bernoulli(p_v_given_h)

  def train(self, v0: torch, vk: torch, ph0: torch, phk: torch):
    """
      Function to train the visible and hidden nodes after k iterations

      Parameters
      ----------
        v0 (torch): the visible nodes
        vk (torch): the visible nodes after the k iterations (sample)
        ph0 (torch): the hidden nodes
        phk (torch): the hidden nodes after the k iterations (sample)

      References
      ----------
        train() - https://www.udemy.com/course/deeplearning/learn/lecture/6895684#overview
    """
    self.W += (torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)).t()
    self.b += torch.sum((v0 - vk), 0)
    self.a += torch.sum((ph0 - phk), 0)