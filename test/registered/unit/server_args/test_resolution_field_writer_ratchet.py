"""Ratchet on §6f Invariant 2 of the field-registry design
(`16-field-registry-design.md`): "the different hooks of one stage must not
write the same field." Two hooks racing to decide the same field with no
ordering contract between them is a real collision, not a normal
model-specific-then-platform-specific override chain.

The design doc's own census of `arg_groups/*_hook.py`'s `declare_resolution`
calls found 20 fields with more than one writer module; this test reproduces
that census fresh against the live tree (so it can't silently go stale) and
ratchets it: today's known collisions are grandfathered by name below, and
the census finding any field outside that set fails loudly, at review time,
instead of being discovered by a scheduler race months later.

Scope, stated rather than assumed (see `census-proxy-vs-target`): this walks
only `arg_groups/*_hook.py` -- the same scope the design doc's own table
used. A few hook steps delegate the bulk of their `declare_resolution` calls
to a helper file outside `arg_groups/` (`handle_npu_backends` to
`hardware_backend/npu/utils.py`, `handle_expert_pack` to
`model_loader/expert_pack_runtime.py`); a collision routed through one of
those would not be caught here. That gap predates this test -- it is why the
doc's own count (253 call sites) already matches this scope, not a wider one.
"""

import ast
import glob
import os
import unittest

import sglang
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_ARG_GROUPS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(sglang.__file__)), "srt", "arg_groups"
)

# Fields already known to have more than one writer module, as of this
# ratchet's introduction -- grandfathered, not endorsed. Each is a real
# same-stage collision per the design doc (§6f), not yet resolved:
#
#   cuda_graph_config                    attention, cuda_graph, dllm, memory,
#                                         model, moe, parallel, platform, serving
#   chunked_prefill_size                 attention, cuda_graph, memory, model,
#                                         parallel
#   disable_radix_cache                  attention, model, pd_disaggregation
#   disable_cuda_graph                   expert_pack, model, parallel
#   moe_a2a_backend                      mega_moe, moe, parallel
#   linear_attn_decode_backend           attention, kimi_k3
#   mem_fraction_static                  attention, memory
#   enable_dp_attention                  deepseek_v4, parallel
#   moe_dense_tp_size                    deepseek_v4, parallel
#   max_running_requests                 deepseek_v4, speculative
#   enable_lora                          dllm, lora
#   enable_mixed_chunk                   dllm, speculative
#   speculative_draft_attention_backend  kimi_k3, speculative
#   enable_lora_overlap_loading          lora, speculative
#   moe_runner_backend                   mega_moe, moe
#   enable_tokenizer_batch_encode        model, serving
#   tokenizer_path                       model_path, serving
#   speculative_draft_model_path         model_path, speculative
#   disable_overlap_schedule             platform, speculative
#
# A field leaving this set (its collision got fixed) should be removed here --
# `test_every_known_collision_is_still_live` catches a whitelist that has
# gone stale in that direction.
_KNOWN_COLLISIONS = frozenset(
    {
        "cuda_graph_config",
        "chunked_prefill_size",
        "disable_radix_cache",
        "disable_cuda_graph",
        "moe_a2a_backend",
        "linear_attn_decode_backend",
        "mem_fraction_static",
        "enable_dp_attention",
        "moe_dense_tp_size",
        "max_running_requests",
        "enable_lora",
        "enable_mixed_chunk",
        "speculative_draft_attention_backend",
        "enable_lora_overlap_loading",
        "moe_runner_backend",
        "enable_tokenizer_batch_encode",
        "tokenizer_path",
        "speculative_draft_model_path",
        "disable_overlap_schedule",
    }
)


def _field_writer_modules() -> dict:
    """field name -> set of `arg_groups/*_hook.py` modules (file stem, minus
    `_hook`) whose `declare_resolution(server_args, source, **fields)` calls
    name it as a keyword. Derived fresh from an AST walk every time -- this
    is the whole point: no hand-kept list to drift from the code."""
    writers: dict = {}
    for path in sorted(glob.glob(os.path.join(_ARG_GROUPS_DIR, "*_hook.py"))):
        module = os.path.basename(path)[: -len("_hook.py")]
        tree = ast.parse(open(path).read())
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "declare_resolution"
            ):
                for kw in node.keywords:
                    if kw.arg is None:
                        continue  # a **mapping forward, not a named field
                    writers.setdefault(kw.arg, set()).add(module)
    return writers


class TestFieldWriterRatchet(CustomTestCase):
    def test_no_new_multi_writer_fields(self):
        writers = _field_writer_modules()
        multi = {field: mods for field, mods in writers.items() if len(mods) > 1}
        unexpected = {
            field: sorted(mods)
            for field, mods in multi.items()
            if field not in _KNOWN_COLLISIONS
        }
        self.assertEqual(
            unexpected,
            {},
            "field(s) newly written by more than one arg_groups/*_hook.py "
            "module (shown with their writer modules) -- either this is a "
            "same-stage collision and one of the writers is wrong, or it's "
            "a deliberate model-specific/platform-specific override chain "
            "that needs a human read before joining _KNOWN_COLLISIONS above",
        )

    def test_every_known_collision_is_still_live(self):
        writers = _field_writer_modules()
        stale = {
            field for field in _KNOWN_COLLISIONS if len(writers.get(field, ())) <= 1
        }
        self.assertEqual(
            stale,
            set(),
            "field(s) in _KNOWN_COLLISIONS no longer have more than one "
            "writer module -- the collision was fixed; remove them from "
            "the whitelist so it doesn't grandfather a problem that no "
            "longer exists",
        )


if __name__ == "__main__":
    unittest.main()
