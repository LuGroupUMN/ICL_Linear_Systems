import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else
                      "mps" if torch.backends.mps.is_available() else
                      "cpu")


class PQModel(nn.Module):
    def __init__(self, d):
        super(PQModel, self).__init__()
        self.P = nn.Parameter(torch.eye(d, device=device) + 0.005 * torch.randn(d, d, device=device))
        self.Q = nn.Parameter(torch.eye(d, device=device) + 0.005 * torch.randn(d, d, device=device))

    def forward(self, A, x):
        mat = self.P @ A @ self.Q
        return torch.bmm(mat, x.unsqueeze(2)).squeeze(2)


class PQModel_far(nn.Module):
    def __init__(self, d):
        super(PQModel_far, self).__init__()
        self.P = nn.Parameter(torch.eye(d, device=device) + 0.1 * torch.randn(d, d, device=device))
        self.Q = nn.Parameter(torch.eye(d, device=device) + 0.1 * torch.randn(d, d, device=device))

    def forward(self, A, x):
        mat = self.P @ A @ self.Q
        return torch.bmm(mat, x.unsqueeze(2)).squeeze(2)



class PQModelop(nn.Module):
    def __init__(self, d):
        super(PQModelop, self).__init__()
        self.P = nn.Parameter(torch.eye(d, device=device) + 0.0 * torch.randn(d, d, device=device))
        self.Q = nn.Parameter(torch.eye(d, device=device) + 0.0 * torch.randn(d, d, device=device))

    def forward(self, A, x):
        mat = self.P @ A @ self.Q
        return torch.bmm(mat, x.unsqueeze(2)).squeeze(2)

class PQModelop_lognormal(nn.Module):
    def __init__(self, d, alpha=2, beta=2):
        super(PQModelop_lognormal, self).__init__()
        self.P = nn.Parameter(torch.eye(d, device=device))
        n = torch.arange(1, d + 1, dtype=torch.float32, device=device)
        theo_cov_diag = ((n * torch.pi) ** 2 + alpha) ** (beta)
        theo_cov = torch.diag(theo_cov_diag)
        self.Q = nn.Parameter(theo_cov) + 0.0001 * torch.randn(d, d, device=device)

    def forward(self, A, x):
        mat = self.P @ A @ self.Q
        return torch.bmm(mat, x.unsqueeze(2)).squeeze(2)


def forward(self, A, x):
        mat = self.P @ A @ self.Q
        return torch.bmm(mat, x.unsqueeze(2)).squeeze(2)


class PQModelbad(nn.Module):
    def __init__(self, d):
        super(PQModelbad, self).__init__()
        p_values = (torch.rand(d, device=device)*10 + 10)
        self.P = nn.Parameter(torch.diag(p_values))
        self.Q = nn.Parameter(torch.diag(1/p_values))

    def forward(self, A, x):
        mat = self.P @ A @ self.Q
        return torch.bmm(mat, x.unsqueeze(2)).squeeze(2)

class PQModelbad2(nn.Module):
    def __init__(self, d):
        super(PQModelbad2, self).__init__()
        p_values = (torch.rand(d, device=device)*2 + 1)
        self.P = nn.Parameter(torch.diag(p_values))
        self.Q = nn.Parameter(torch.diag(1/p_values))

    def forward(self, A, x):
        mat = self.P @ A @ self.Q
        return torch.bmm(mat, x.unsqueeze(2)).squeeze(2)

class PQModelbad3(nn.Module):
    def __init__(self, d):
        super(PQModelbad3, self).__init__()
        M = torch.randn(d, d, device=device)
        M_inv = torch.inverse(M)
        self.P = nn.Parameter(M)
        self.Q = nn.Parameter(M_inv)

    def forward(self, A, x):
        mat = self.P @ A @ self.Q
        return torch.bmm(mat, x.unsqueeze(2)).squeeze(2)

class PQModelop_testQn(nn.Module):
    def __init__(self, d, n, c=2):
        super(PQModelop_testQn, self).__init__()
        k = torch.arange(1, d + 1).to(device)
        delta_mat = torch.diag(k**2) / (d+1)**1.0
        self.P = nn.Parameter(torch.eye(d, device=device))
        self.Q = nn.Parameter(torch.eye(d, device=device) + 1/n * c* torch.randn(d, d, device=device))
        #self.Q = nn.Parameter(torch.eye(d, device=device) + 1 / n * delta_mat)

    def forward(self, A, x):
        mat = self.P @ A @ self.Q
        return torch.bmm(mat, x.unsqueeze(2)).squeeze(2)