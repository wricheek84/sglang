import unittest
from unittest.mock import MagicMock, patch

from sglang.srt.arg_groups.lora_hook import check_lora_speculative_compatibility
from sglang.test.ci_registration import register_cpu_ci


class TestLoRASpecAlgoGate(unittest.TestCase):
    def setUp(self):
        self.mock_server_args = MagicMock()

        self.patcher_resolving_view = patch("sglang.srt.arg_groups.lora_hook.resolving_view")
        self.mock_resolving_view = self.patcher_resolving_view.start()

        self.mock_cfg = MagicMock()
        self.mock_cfg.speculative_adaptive = False
        self.mock_cfg.moe_runner_backend = "default"
        self.mock_cfg.speculative_moe_runner_backend = "default"
        self.mock_resolving_view.return_value = self.mock_cfg

        self.patcher_envs = patch("sglang.srt.arg_groups.lora_hook.envs")
        self.mock_envs = self.patcher_envs.start()
        self.mock_envs.SGLANG_RAGGED_VERIFY_MODE.get.return_value = "static"
        self.mock_envs.SGLANG_ENABLE_OVERLAP_PLAN_STREAM.get.return_value = False

    def tearDown(self):
        self.patcher_resolving_view.stop()
        self.patcher_envs.stop()

    def test_supported_algorithms_case_insensitive(self):
        """Scenario 1: Valid algorithms should pass regardless of casing."""
        valid_algos = ["EAGLE", "eagle", "Eagle", "DSPARK", "dspark", "Dspark", "DFLASH", "dflash", "EAGLE3", "eagle3"]

        for algo in valid_algos:
            with self.subTest(algo=algo):
                self.mock_cfg.speculative_algorithm = algo
                try:
                    check_lora_speculative_compatibility(self.mock_server_args)
                except ValueError as e:
                    self.fail(f"Valid algorithm {algo} improperly raised ValueError: {e}")

    def test_none_and_ngram_early_exit(self):
        """Scenario 2: None and NGRAM (any casing) should exit early without error."""
        early_exit_cases = [None, "NGRAM", "ngram", "Ngram"]

        for algo in early_exit_cases:
            with self.subTest(algo=algo):
                self.mock_cfg.speculative_algorithm = algo
                try:
                    check_lora_speculative_compatibility(self.mock_server_args)
                except Exception as e:
                    self.fail(f"{algo} should exit early but raised: {e}")

    def test_unsupported_algorithm_preserves_raw_input(self):
        """Scenario 3: Unsupported algorithms raise, and the message echoes the
        raw input as given — not the uppercased internal form. This guards
        against leaking the normalization detail into user-facing output."""
        cases = ["my_custom_draft", "MY_CUSTOM_DRAFT"]

        for algo in cases:
            with self.subTest(algo=algo):
                self.mock_cfg.speculative_algorithm = algo
                with self.assertRaises(ValueError) as context:
                    check_lora_speculative_compatibility(self.mock_server_args)
                self.assertIn(f"not {algo}.", str(context.exception))

    def test_frozen_kv_mtp_hint(self):
        """Scenario 4: The FROZEN_KV_MTP hint should trigger case-insensitively."""
        for algo in ["FROZEN_KV_MTP", "frozen_kv_mtp"]:
            with self.subTest(algo=algo):
                self.mock_cfg.speculative_algorithm = algo
                with self.assertRaises(ValueError) as context:
                    check_lora_speculative_compatibility(self.mock_server_args)
                self.assertIn("automatically promoted to FROZEN_KV_MTP", str(context.exception))

    def test_dspark_ragged_verify_guard(self):
        """Scenario 5: DSPARK ragged verify guard must trigger case-insensitively."""
        self.mock_envs.SGLANG_RAGGED_VERIFY_MODE.get.return_value = "dynamic"

        for algo in ["DSPARK", "dspark"]:
            with self.subTest(algo=algo):
                self.mock_cfg.speculative_algorithm = algo
                with self.assertRaises(ValueError) as context:
                    check_lora_speculative_compatibility(self.mock_server_args)
                self.assertIn("does not support SGLANG_RAGGED_VERIFY_MODE='dynamic'", str(context.exception))

    def test_ragged_verify_guard_is_dspark_specific(self):
        """Scenario 6: A dynamic ragged mode must NOT trip up the other
        lora-compatible algorithms. Without this test, the string-equality
        check on `speculative_algorithm == "DSPARK"` could be loosened or
        broken (e.g. to an `in` check against the whole tuple) and nothing
        here would catch it."""
        self.mock_envs.SGLANG_RAGGED_VERIFY_MODE.get.return_value = "dynamic"

        for algo in ["EAGLE", "eagle", "EAGLE3", "DFLASH", "dflash"]:
            with self.subTest(algo=algo):
                self.mock_cfg.speculative_algorithm = algo
                try:
                    check_lora_speculative_compatibility(self.mock_server_args)
                except ValueError as e:
                    self.fail(f"{algo} should be unaffected by dynamic ragged mode but raised: {e}")

    def test_overlap_plan_stream_guard(self):
        """Scenario 7: SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1 must reject all 
        supported algorithms because batch prep runs unordered."""
        self.mock_envs.SGLANG_ENABLE_OVERLAP_PLAN_STREAM.get.return_value = True
        
        for algo in ["EAGLE", "eagle", "DSPARK"]:
            with self.subTest(algo=algo):
                self.mock_cfg.speculative_algorithm = algo
                with self.assertRaises(ValueError) as context:
                    check_lora_speculative_compatibility(self.mock_server_args)
                
                self.assertIn("does not support SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1", str(context.exception))


if __name__ == "__main__":
    register_cpu_ci(est_time=5, suite="base-a-test-cpu")
    unittest.main()