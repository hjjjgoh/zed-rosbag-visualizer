# export_fs_scripted.py
import torch
from pathlib import Path
from src.preprocess.sensor_module.foundation_stereo.foundation_stereo import FoundationStereo
from src.preprocess.sensor_module.stereo_estimator import AttrDict

WEIGHTS = "models/model_best_bp2.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ScriptableFS(torch.nn.Module):
    def __init__(self, core: torch.nn.Module, iters: int = 12):
        super().__init__()
        self.core = core
        self.iters = iters

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        # 단일 텐서 출력으로 고정 (test_mode=True)
        disp_up = self.core(left, right, iters=self.iters, test_mode=True, low_memory=False)
        return disp_up  # [B,1,H,W]

def build_model() -> torch.nn.Module:
    args = AttrDict(
        max_disp=416,
        mixed_precision=False,          # 스크립팅 안정성 위해 OFF 권장
        hidden_dims=[128,128,128],
        n_downsample=2,
        n_gru_layers=3,
        slow_fast_gru=False,
        corr_levels=2,
        corr_radius=4,
        context_norm='batch',
        vit_size='vits',
        corr_implementation='reg',
        low_memory=False,
    )
    model = FoundationStereo(args)
    ckpt = torch.load(WEIGHTS, map_location="cpu")
    state_dict = ckpt.get("model") or ckpt.get("state_dict") or ckpt
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.to(DEVICE).eval()
    # 안전: 스크립팅 시 autocast 영향 차단
    if hasattr(model, "args"):
        model.args.mixed_precision = False
    return model

def main():
    core = build_model()
    wrapped = ScriptableFS(core, iters=12).to(DEVICE).eval()

    # 32의 배수 dummy (모델 내부 pad와 호환)
    H, W = 736, 1280
    left  = torch.randn(1, 3, H, W, device=DEVICE)
    right = torch.randn(1, 3, H, W, device=DEVICE)

    with torch.inference_mode():
        # 기준 출력
        ref = wrapped(left, right)

        # ✅ trace 시 입력 예시 지정 (필수)
        scripted = torch.jit.trace(wrapped, (left, right))

        Path("models").mkdir(parents=True, exist_ok=True)
        scripted_path = "models/foundation_stereo_scripted.pt"
        scripted.save(scripted_path)
        print(f"[export] Saved TorchScript: {scripted_path}")

        # 로드 & 검증
        loaded = torch.jit.load(scripted_path, map_location=DEVICE)
        out = loaded(left, right)
        mae = (ref - out).abs().mean().item()
        print(f"[check] Ref vs Scripted MAE: {mae:.6f}")


if __name__ == "__main__":
    main()
