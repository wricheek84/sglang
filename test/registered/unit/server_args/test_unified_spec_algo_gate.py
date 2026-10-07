# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""`--enable-unified-memory` speculative algorithm compatibility gate.

Guards that unified memory accepts only DSPARK (with linear topk=1 or None)
regardless of casing ("DSPARK", "dspark", "Dspark"), safely accepts None,
and rejects unsupported algorithms and invalid topk settings.

    python -m pytest test/registered/unit/server_args/test_unified_spec_algo_gate.py -v
"""

import unittest
from unittest.mock import patch

import msgspec

from sglang.srt.arg_groups.kv_cache_hook import handle_unified_memory_pool
from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _run_handler(
    *,
    unified=True,
    spec_algo=None,
    topk=None,
    attention_backends=("triton", "triton"),
):
    """Run just `handle_unified_memory_pool` over a minimal stand-in."""
    sa = ServerArgs(model_path="dummy")
    for name, value in {
        "enable_unified_memory": unified,
        "disaggregation_mode": "null",
        "speculative_algorithm": spec_algo,
        "speculative_eagle_topk": topk,
        "enable_hierarchical_cache": False,
        "enable_lmcache": False,
        "enable_two_batch_overlap": False,
        "dcp_size": 1,
        "cuda_graph_config": None,
        "cuda_graph_backend_prefill": Backend.DISABLED,
    }.items():
        msgspec.Struct.__setattr__(sa, name, value)
    with patch(
        "sglang.srt.arg_groups.kv_cache_hook.attention_backends_of",
        return_value=attention_backends,
    ):
        handle_unified_memory_pool(sa)


class TestUnifiedSpecAlgoGate(unittest.TestCase):
    def test_dspark_casing_variations_accepted(self):
        """Unified memory must accept DSPARK across all casing variants."""
        for algo in ("DSPARK", "dspark", "Dspark"):
            with self.subTest(spec_algo=algo):
                _run_handler(unified=True, spec_algo=algo, topk=1)

    def test_none_spec_algo_accepted(self):
        """Default non-speculative runs must pass cleanly."""
        _run_handler(unified=True, spec_algo=None)

    def test_unsupported_spec_algo_rejected(self):
        """Algorithms other than DSPARK must be rejected."""
        for algo in ("EAGLE", "eagle", "NEXTN", "ngram"):
            with self.subTest(spec_algo=algo):
                with self.assertRaises(AssertionError) as ctx:
                    _run_handler(unified=True, spec_algo=algo)
                self.assertIn(
                    "only supports --speculative-algorithm DSPARK", str(ctx.exception)
                )

    def test_dspark_invalid_topk_rejected(self):
        """DSPARK under unified memory requires linear draft chain (topk in {None, 1})."""
        for algo in ("DSPARK", "dspark"):
            with self.subTest(spec_algo=algo):
                with self.assertRaises(AssertionError) as ctx:
                    _run_handler(unified=True, spec_algo=algo, topk=2)
                self.assertIn("supports a linear draft chain only", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
