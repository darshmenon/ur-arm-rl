"""Runtime compatibility shims for local training scripts."""

import sys


def block_real_tensorflow():
    """Stop SB3's tensorboard integration from dragging in real TensorFlow.

    This machine's Python env has both TensorFlow (from an unrelated project)
    and Triton installed alongside PyTorch. Importing stable_baselines3 pulls
    in TensorFlow transitively via torch.utils.tensorboard/tensorboard; once
    that's loaded, the first Adam optimizer construction triggers torch.optim's
    lazy `import torch._dynamo`, which dlopens Triton's bundled LLVM -- and the
    two native library stacks corrupt the heap, segfaulting inside
    libtriton.so. None of these training scripts use TensorFlow, so stub it
    out of sys.modules before anything downstream can import it.
    """
    sys.modules.setdefault("tensorflow", None)


def patch_torch_dynamo_for_optimizer():
    """Avoid importing Torch Dynamo/Triton while constructing optimizers.

    The local PyTorch 2.12.0+cu130 / Triton 3.7.0 install can segfault when
    torch.optim lazily imports torch._dynamo. The training scripts do not use
    torch.compile, so bypassing the internal Dynamo wrapper keeps SAC optimizer
    construction on the regular eager path.
    """
    try:
        import torch
        import torch._compile
    except Exception:
        return

    def identity_disable(fn=None, recursive=True):
        if fn is None:
            return lambda wrapped: wrapped
        return fn

    torch._disable_dynamo = identity_disable
    torch._compile._disable_dynamo = identity_disable
