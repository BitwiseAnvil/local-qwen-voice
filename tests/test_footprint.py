"""Host offload and int8 weights on real (CPU) torch modules; skipped where torch is absent.

Run inside the image without a GPU:
docker run --rm -v "${PWD}:/src:ro" -w /src --entrypoint python local-qwen-voice-tts -m unittest discover -s tests
"""

import unittest

try:
    import torch
    from torch import nn
except ImportError:
    torch = None

from local_voice.engine import host_resident_parameters, move_tensors, tensor_slots


@unittest.skipIf(torch is None, "torch is not installed")
class Int8Tests(unittest.TestCase):
    def setUp(self):
        from local_voice import footprint

        self.footprint = footprint
        torch.manual_seed(0)

    def block(self):
        return nn.ModuleDict({
            "attention": nn.Linear(256, 128, bias=False),
            "mlp": nn.Sequential(nn.Linear(256, 384, bias=True), nn.SiLU(), nn.Linear(384, 256, bias=False)),
            "norm": nn.LayerNorm(256),
        }).to(torch.bfloat16)

    def relative_error(self, actual, expected):
        return ((actual.float() - expected.float()).norm() / expected.float().norm()).item()

    def test_int8_linear_matches_bf16(self):
        linear = nn.Linear(256, 384, bias=True).to(torch.bfloat16)
        layer = self.footprint.Int8Linear(linear)
        self.assertEqual(torch.int8, layer.qweight.dtype)
        self.assertEqual((384,), tuple(layer.scale.shape))
        for rows in (1, 24):
            with self.subTest(rows=rows):
                x = torch.randn(2, rows, 256, dtype=torch.bfloat16)
                with torch.no_grad():
                    self.assertLess(self.relative_error(layer(x), linear(x)), 0.02)

    def test_triton_kernel_is_only_enabled_where_it_runs(self):
        layer = self.footprint.Int8Linear(nn.Linear(512, 64, bias=False).to(torch.bfloat16))
        # CPU tensors never take the Triton path.
        self.assertFalse(self.footprint._gemv_ok(layer))
        if not torch.cuda.is_available() or self.footprint.triton is None:
            self.skipTest("needs CUDA and Triton")
        layer = layer.cuda()
        self.assertTrue(self.footprint._gemv_ok(layer))
        layer.gemv = True
        x = torch.randn(1, 1, 512, device="cuda", dtype=torch.bfloat16)
        reference = torch.nn.functional.linear(x.float(), layer.dequantized(torch.float32))
        self.assertLess(self.relative_error(layer(x), reference), 0.01)

    def test_quantize_replaces_only_linears_and_survives_a_park_round_trip(self):
        block = self.block()
        reference = self.block()
        reference.load_state_dict(block.state_dict())
        count, freed, _ = self.footprint.quantize_linears(block)
        self.assertEqual(3, count)
        self.assertGreater(freed, 0)
        self.assertIsInstance(block["norm"], nn.LayerNorm)
        self.assertIsInstance(block["mlp"][0], self.footprint.Int8Linear)
        self.assertIsNotNone(block["mlp"][0].bias)
        x = torch.randn(1, 5, 256, dtype=torch.bfloat16)

        def run(module):
            with torch.no_grad():
                return module["mlp"](module["norm"](x)) + module["attention"](x).sum()

        expected = run(reference)
        self.assertLess(self.relative_error(run(block), expected), 0.03)
        # Idle parking moves int8 buffers like any other: out and back again.
        slots = tensor_slots([block], "cpu")
        names = {name for _, name, _ in slots}
        self.assertTrue({"qweight", "scale", "bias", "weight"} <= names)
        move_tensors(slots, lambda tensor: tensor.clone())
        self.assertEqual(torch.int8, block["attention"].qweight.dtype)
        self.assertLess(self.relative_error(run(block), expected), 0.03)

    def test_host_embedding_matches_and_is_invisible_to_parking(self):
        talker = nn.Module()
        talker.text_embedding = nn.Embedding(1000, 64).to(torch.bfloat16)
        ids = torch.tensor([[1, 5, 999, 5]])
        expected = talker.text_embedding(ids)
        freed = self.footprint.offload_text_embedding(talker, "cpu")
        self.assertEqual(1000 * 64 * 2, freed)
        self.assertTrue(torch.equal(expected, talker.text_embedding(ids)))
        self.assertEqual([], list(talker.parameters()))
        self.assertEqual([], tensor_slots([talker], "cpu"))
        self.assertEqual(0, self.footprint.offload_text_embedding(talker, "cpu"))

    def test_encoder_waits_in_host_ram_and_encodes_on_the_device(self):
        class Encoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Linear(8, 4)
                self.devices = []

            def encode(self, values):
                self.devices.append(self.proj.weight.device.type)
                return self.proj(values)

        tokenizer = nn.Module()
        tokenizer.encoder = Encoder()
        tokenizer.decoder = nn.Linear(4, 8)
        freed = self.footprint.offload_encoder(tokenizer, "cpu")
        self.assertEqual((8 * 4 + 4) * 4, freed)
        self.assertTrue(tokenizer.encoder.host_resident)
        with torch.inference_mode():
            out = tokenizer.encoder.encode(torch.ones(1, 8))
        self.assertEqual((1, 4), tuple(out.shape))
        self.assertEqual(["cpu"], tokenizer.encoder.devices)
        # Parameters stay normal tensors after moves made during inference.
        self.assertFalse(tokenizer.encoder.proj.weight.is_inference())
        self.assertEqual({id(p) for p in tokenizer.encoder.parameters()}, host_resident_parameters(tokenizer))
        self.assertEqual(0, self.footprint.offload_encoder(tokenizer, "cpu"))


if __name__ == "__main__":
    unittest.main()
