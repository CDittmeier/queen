import numpy as np
import pytest
import torch

from mac_inference.mps import CachedCrossAttention
from models.flamingo import DenseXAttn


@pytest.fixture
def layer_and_inputs():
    torch.manual_seed(42)
    layer = DenseXAttn(4, 8, n_heads=2, n_kv_heads=2, wo_rand_init=True).eval()
    return layer, torch.randn(1, 5, 8), torch.randn(1, 3, 4)


def test_cached_board_attention_matches_upstream(layer_and_inputs):
    layer, hidden, board = layer_and_inputs
    with torch.inference_mode():
        cached = CachedCrossAttention(layer, board)
        torch.testing.assert_close(cached(hidden), layer(hidden, board))
        torch.testing.assert_close(cached(hidden[:, -1:]), layer(hidden[:, -1:], board))


def test_mlx_bridge_matches_pytorch_and_replaces_board_cache(layer_and_inputs):
    mx = pytest.importorskip("mlx.core")
    from mac_inference.mlx import CrossAttention

    layer, hidden, board = layer_and_inputs
    bridge = CrossAttention(4, 8, n_heads=2)
    bridge.load_weights(
        [(name, mx.array(value.numpy())) for name, value in layer.state_dict().items()],
        strict=True,
    )
    bridge.eval()
    with torch.inference_mode():
        for states in (board, board + torch.randn_like(board)):
            bridge.set_board(mx.array(states.numpy()))
            actual = np.array(bridge(mx.array(hidden.numpy())))
            expected = layer(hidden, states).numpy()
            np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-5)
