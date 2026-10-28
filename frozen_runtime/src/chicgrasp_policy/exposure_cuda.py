"""CUDA-only synthetic encoder shape inference without changing initialization RNG."""
from contextlib import contextmanager
from . import overfit_policy as policy_api


@contextmanager
def cuda_constructor(receipts):
    import torch
    policy_api._enable_official_imports()
    from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
    if not torch.cuda.is_available():raise RuntimeError('CUDA constructor required')
    original=MultiImageObsEncoder.output_shape
    def output_shape(encoder):
        # Weight initialization still uses the original CPU generator. Only the
        # existing synthetic shape forward moves to CUDA; no default-device edit.
        encoder.to('cuda:0')
        before_cpu=torch.get_rng_state().clone()
        before_cuda=torch.cuda.get_rng_state(0).clone()
        before_state={key:value.detach().clone() for key,value in encoder.state_dict().items()}
        shape=original(encoder)
        rng_unchanged=(torch.equal(before_cpu,torch.get_rng_state()) and
                       torch.equal(before_cuda,torch.cuda.get_rng_state(0)))
        state_unchanged=all(torch.equal(value,encoder.state_dict()[key]) for key,value in before_state.items())
        if not rng_unchanged or not state_unchanged:
            raise RuntimeError('Constructor shape inference changed RNG or encoder state')
        receipts.append(dict(operation='existing synthetic encoder.output_shape',device=str(encoder.device),
            output_shape=list(shape),rng_unchanged=True,encoder_state_unchanged=True,
            initialization='Original CPU weight initialization retained; no CPU model forward'))
        return shape
    MultiImageObsEncoder.output_shape=output_shape
    try:yield
    finally:MultiImageObsEncoder.output_shape=original
