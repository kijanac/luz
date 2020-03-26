import luz
import numpy as np
import torch

def test_predictor_trivial():
    x = torch.tensor([0.])

    fc = luz.Predictor(torch.nn.Linear(in_features=1, out_features=1,bias=False))

    np.testing.assert_allclose(fc.predict(x).detach().numpy(),np.array([0.]))