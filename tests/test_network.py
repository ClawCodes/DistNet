import pytest
import torch
from distnet.localnet import LocalNet, DistLocalNet
from distnet.util import load_mnist, train, test, set_seed

def assert_models_equivalent(model1, model2, tol=1e-6):
    for (n1, p1), (n2, p2) in zip(model1.named_parameters(), model2.named_parameters()):
        assert n1 == n2
        assert torch.allclose(p1, p2, atol=tol), f"Param {n1} differs!"


class TestDistNetworks:
    def test_params_align_and_average(self):
        # Create identical instances
        model1 = LocalNet()
        model2 = LocalNet()

        params1 = list(model1.parameters())
        params2 = list(model2.parameters())

        # Parameters should be registered and fetched in deterministic order
        assert len(params1) == len(params2), "Parameter count mismatch"

        #
        for i, (p1, p2) in enumerate(zip(params1, params2)):
            assert p1.shape == p2.shape, f"Shape mismatch at param {i}"

            # simulate local gradients
            g1 = torch.full_like(p1, 1.0)  # Create tensor of p1 shape filled with 1.0
            g2 = torch.full_like(p2, 2.0)  # Create tensor of p2 shape filled with 2.0

            # simulate all-reduce average
            avg = 0.5 * (g1 + g2)

            # Tensor of shape p1 filled with 1.5 is the expected average of g1 and g2
            expected = torch.full_like(p1, 1.5)
            assert torch.allclose(avg, expected), f"Grad mismatch at param {i}"

    def test_dist_hook_with_no_grad_change(self):
        model1 = LocalNet().to(torch.device('cpu'))
        model2 = DistLocalNet().to(torch.device('cpu')) # LocalNet that inherits from DistNet

        model2.load_state_dict(model1.state_dict()) # ensure same initialization

        def dist_hook(name, grad):
            return grad # return the same gradient so training should be identical

        model2.register_grad_hook(dist_hook)

        train_loader, test_loader = load_mnist(shuffle=False)

        set_seed()
        train(model1, train_loader, epochs=1)

        set_seed()
        train(model2, train_loader, epochs=1)

        assert_models_equivalent(model1, model2)

    def test_dist_hook_with_grad_change(self):
        model1 = LocalNet().to(torch.device('cpu'))
        model2 = DistLocalNet().to(torch.device('cpu')) # LocalNet that inherits from DistNet

        model2.load_state_dict(model1.state_dict()) # ensure same initialization

        def dist_hook(name, grad):
            return grad * 2 # multiply gradient by 2 to alter gradient

        model2.register_grad_hook(dist_hook)

        train_loader, test_loader = load_mnist(shuffle=False)

        set_seed()
        train(model1, train_loader, epochs=1)

        set_seed()
        train(model2, train_loader, epochs=1)

        with pytest.raises(AssertionError, match="Param .* differs"):
            assert_models_equivalent(model1, model2)
    def test_dist_hook_with_all_reduce(self):
        #Testing to make sure dist hook works with custom all reduce hook + making sure bucketing params doesn't break anything
        model1 = LocalNet().to(torch.device('cpu'))
        model2 = DistLocalNet().to(torch.device('cpu')) # LocalNet that inherits from DistNet

        model2.load_state_dict(model1.state_dict()) # ensure same initialization

        model2.register_grad_hook(model2.dist_hook)

        train_loader, test_loader = load_mnist(shuffle=False)

        set_seed()
        train(model1, train_loader, epochs=1)

        set_seed()
        train(model2, train_loader, epochs=1)

        assert_models_equivalent(model1, model2)