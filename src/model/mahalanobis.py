# Add this cell at the top of the notebook
from copy import deepcopy
from typing import List, Tuple, Callable

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader

from sklearn.covariance import LedoitWolf
from tqdm import tqdm


class PoolingMahalanobisDetector(nn.Module):
    def __init__(
        self,
        swivel: str,
        pool: str = 'avg2d',
        sigma_algorithm: str = 'default',
        sigma_diag_eps: float = 2e-1,
        transform: bool = False,
        dist_fn: str = 'squared_mahalanobis',
        lr: float = 1e-3,
        device: str  = 'cuda:0'
    ):
        super().__init__()
        # init args
        self.swivel = swivel
        self.pool = pool
        self.sigma_algorithm = sigma_algorithm
        self.sigma_diag_eps = sigma_diag_eps
        # self.hook_fn = self.register_forward_pre_hook if hook_fn == 'pre' else self.register_forward_hook
        self.dist_fn = dist_fn
        self.lr = lr
        self.device = device
        # other attributes
        self.active = True
        self.training_representations = []
        # methods
        if self.pool == 'avg2d':
            self._pool = nn.AvgPool2d(
                kernel_size=(2,2), 
                stride=(2,2)
            )
        elif self.pool == 'avg3d':
            self._pool = nn.AvgPool3d(
                kernel_size=(2,2,2),
                stride=(2,2,2)
            )
        elif self.pool == 'none':
            self._pool = None
        else:
            raise NotImplementedError('Choose from: avg2d, avg3d, none')
        
        self.to(device)


    ### private methods ###

    def _reduce(self, x: Tensor) -> Tensor:
        if 'avg' in self.pool:
            # reduce dimensionality with 3D pooling to below 1e4 entries
            while torch.prod(torch.tensor(x.shape[1:])) > 1e4:
                x = self._pool(x)
            x = self._pool(x)
        elif self.pool == 'none':
            pass
        # reshape to (batch_size, 1, n_features)
        x = x.reshape(x.shape[0], 1, -1)
        return x


    @torch.no_grad()
    def _collect(self, x: Tensor) -> None:
        # reduces dimensionality as per self._pool, moves to cpu and stores
        x = self._reduce(x.detach()).cpu()
        self.training_representations.append(x)


    @torch.no_grad()
    def _merge(self) -> None:
        # concatenate batches from training data
        self.training_representations = torch.cat(
            self.training_representations,
            dim=0
        )


    @torch.no_grad()
    def _estimate_gaussians(self) -> None:
        self.register_buffer(
            'mu',
            self.training_representations.mean(0, keepdims=True).detach().to(self.device)
        )
        
        if self.sigma_algorithm == 'diagonal':
            self.register_buffer(
                'var',
                torch.var(self.training_representations.squeeze(1), dim=0).detach()
            )
            sigma = torch.sqrt(self.var)
            sigma = torch.max(sigma, torch.tensor(self.sigma_diag_eps))
            self.register_buffer(
                'sigma_inv', 
                1 / sigma.detach().to(self.device)
            )

            # self.sigma_inv = torch.max(self.sigma_inv, torch.tensor(self.sigma_diag_eps).to(self.device))
        
        elif self.sigma_algorithm == 'default':
            assert self.pool in ['avg2d', 'avg3d'], 'default only works with actual pooling, otherwise calculation sigma is infeasible'

            self.register_buffer(
                'sigma',
                torch.cov(self.training_representations.squeeze(1).T)
            )
            self.register_buffer(
                'sigma_inv', 
                torch.linalg.inv(self.sigma).detach().unsqueeze(0).to(self.device)
            )

        elif self.sigma_algorithm == 'ledoit_wolf':
            assert self.pool in ['avg2d', 'avg3d'], 'default only works with actual pooling, otherwise calculation sigma is infeasible'

            self.register_buffer(
                'sigma', 
                torch.from_numpy(
                    LedoitWolf().fit(
                        self.training_representations.squeeze(1)
                    ).covariance_
                )
            )
            self.register_buffer(
                'sigma_inv', 
                torch.linalg.inv(self.sigma).detach().unsqueeze(0).to(self.device)
            )

        else:
            raise NotImplementedError('Choose from: ledoit_wolf, diagonal, default')


    def _distance(self, x: Tensor) -> Tensor:
        assert self.sigma_inv is not None, 'fit the model first'
        x_reduced  = self._reduce(x)
        x_centered = x_reduced - self.mu
        if self.sigma_algorithm == 'diagonal':
            dist = x_centered**2 * self.sigma_inv
            dist = dist.sum((1,2))
        else:
            dist = x_centered @ self.sigma_inv @ x_centered.permute(0,2,1)
        dist = dist.view((dist.shape[0], ))
        assert len(dist.shape) == 1, 'distances should be 1D over batch dimension'
        assert dist.shape[0] == x.shape[0], 'distance and input should have same batch size'

        if self.dist_fn == 'squared_mahalanobis':
            return dist
        elif self.dist_fn == 'mahalanobis':
            return torch.sqrt(dist)
        else:
            raise NotImplementedError('Choose from: squared_mahalanobis, mahalanobis')


    ### public methods ###

    def fit(self):
        self._merge()
        self._estimate_gaussians()
        # del self.training_representations


    def on(self):
        self.active = True


    def off(self):
        self.active = False


    def forward(self, x: Tensor) -> Tensor:
        if self.active:
            if self.training:
                self._collect(x)
            else:
                self.batch_distances = self._distance(x).detach().view(-1)
        else:
            pass
        return x
    

class PoolingMahalanobisWrapper(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        adapters: nn.ModuleList,
        copy: bool = True,
    ):
        super().__init__()
        self.model           = deepcopy(model) if copy else model
        self.adapters        = adapters
        self.adapter_handles = {}
        self.model.eval()


    def hook_adapters(
        self,
    ) -> None:
        for adapter in self.adapters:
            swivel = adapter.swivel
            layer  = self.model.get_submodule(swivel)
            hook   = self._get_hook(adapter)
            self.adapter_handles[
                swivel
            ] = layer.register_forward_pre_hook(hook)


    def _get_hook(
        self,
        adapter: nn.Module
    ) -> Callable:
        def hook_fn(
            module: nn.Module, 
            x: Tuple[Tensor]
        ) -> Tensor:
            # x, *_ = x # tuple, alternatively use x_in = x[0]
            # x = adapter(x)
            return adapter(x[0])
        
        return hook_fn
    

    def fit(self):
        for adapter in self.adapters:
            adapter.fit()


    def set_lr(self, lr: float):
        for adapter in self.adapters:
            adapter.lr = lr


    def turn_off_all_adapters(self):
        for adapter in self.adapters:
            adapter.off()


    def forward(
        self, 
        x: Tensor
    ) -> Tensor:
        return self.model(x).detach().cpu()
    


def get_pooling_mahalanobis_detector(
    swivels: List[str],
    unet: nn.Module = None,
    pool: str = 'avg2d',
    sigma_algorithm: str = 'default',
    dist_fn: str = 'squared_mahalanobis',
    fit: str  = 'raw', # None, 'raw', 'augmented'
    iid_data: DataLoader = None,
    transform: bool = False,
    lr: float = 1e-3,
    device: str  = 'cuda:0',
): 

    pooling_detector = [
        PoolingMahalanobisDetector(
            swivel=swivel,
            device=device,
            pool=pool,
            sigma_algorithm=sigma_algorithm,
            transform=transform,
            dist_fn=dist_fn,
            lr=lr,
        ) for swivel in swivels
    ]
    pooling_wrapper = PoolingMahalanobisWrapper(
        model=unet,
        adapters=nn.ModuleList(pooling_detector),
        copy=True,
    )
    pooling_wrapper.hook_adapters()
    pooling_wrapper.to(device)
    if fit == 'raw':
        for batch in iid_data:
            x = batch['input'].to(device)
            _ = pooling_wrapper(x)
    elif fit == 'augmented':
        for _ in range(250):
            batch = next(iid_data)
            x = batch['data'].to(device)
            _ = pooling_wrapper(x)
    if fit is not None:
        pooling_wrapper.fit()
        pooling_wrapper.eval()

    return pooling_wrapper