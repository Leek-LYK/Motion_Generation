"""Tiny x0 diffusion training, exact continuation, and independent DDIM sampling."""
import copy
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from kimodo.model.diffusion import DDIMSampler, Diffusion
from kimodo.model.twostage_denoiser import TwostageDenoiser
from kimodo.motion_rep import KimodoMotionRep
from kimodo.skeleton.registry import build_skeleton

from data_pipeline.model_input_dataset import G1ModelInputDataset, collate_model_inputs

ROOT = Path(__file__).resolve().parents[1]
INPUTS = ROOT / "data/processed/g1_model_inputs_v1"
COMPONENTS = {"root_pos": (0, 3), "heading": (3, 5), "joint_pos": (5, 107),
              "joint_rot": (107, 311), "joint_vel": (311, 413), "contacts": (413, 417)}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(tmp, path)


def configure(seed, threads=4):
    torch.set_num_threads(threads)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.mha.set_fastpath_enabled(False)
    torch.use_deterministic_algorithms(True)


def validate_inputs():
    ready = json.loads((INPUTS / "READY").read_text())
    conf = json.loads((INPUTS / "config.json").read_text())
    if not ready["passed"] or ready["fingerprint"] != conf["fingerprint"]:
        raise ValueError("M2 READY/config mismatch")
    if sha(ROOT / "data_pipeline/model_input_dataset.py") != ready["dataset_sha256"]:
        raise ValueError("Dataset changed after M2 validation")
    audit = json.loads((INPUTS / "stats/independent_audit.json").read_text())
    for path, expected in audit["stats_files_sha256"].items():
        if sha(INPUTS / path) != expected:
            raise ValueError(f"M2 statistics changed: {path}")
    for path, expected in conf["source_sha256"].items():
        if sha(ROOT.parent / "kimodo" / path) != expected:
            raise ValueError(f"Kimodo source changed: {path}")
    text = json.loads((INPUTS / "text_encoder/COMPLETE").read_text())
    if text["fingerprint"] != ready["text_config_fingerprint"]:
        raise ValueError("Text cache fingerprint mismatch")
    return ready


def load_samples(manifest, split):
    entries = manifest[split]
    dataset = G1ModelInputDataset(split)
    wanted = {x["sample_id"]: x for x in entries}
    by_id = {r["sample_id"]: i for i, r in enumerate(dataset.rows) if r["sample_id"] in wanted}
    if set(by_id) != set(wanted):
        raise ValueError(f"Missing subset samples in {split}")
    samples = []
    for entry in entries:
        sample = dataset[by_id[entry["sample_id"]]]
        for key in ("text_id", "text", "motion_id"):
            if sample[key] != entry[key]:
                raise ValueError(f"Subset {key} changed")
        sample["group"] = entry["group"]
        samples.append(sample)
    return samples


def make_model(config, stats_dir=None):
    rep = KimodoMotionRep(build_skeleton(34), fps=30,
                          stats_path=str(stats_dir or INPUTS / "stats/motion"))
    return TwostageDenoiser(
        rep, motion_mask_mode="concat", ckpt_path=None, llm_shape=[1, 4096],
        use_text_mask=False, activation="gelu", dropout=0.0, pe_dropout=0.0,
        norm_first=False, num_text_tokens_override=50, input_first_heading_angle=True,
        **config,
    ).float()


def batch_to(samples, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in collate_model_inputs(samples).items()}


def forward(model, x, batch, t):
    return model(x, batch["x_pad_mask"], batch["text_feat"], batch["text_feat_pad_mask"], t,
                 first_heading_angle=batch["first_heading_angle"])


def losses(prediction, target, mask):
    error = torch.nn.functional.smooth_l1_loss(prediction, target, reduction="none")
    valid = mask[..., None]
    values = {name: (error[..., a:b] * valid).sum() / (mask.sum() * (b - a))
              for name, (a, b) in COMPONENTS.items()}
    # Debug objective: mean of six normalized component means, not the paper's physical/FK loss.
    total = torch.stack(list(values.values())).mean()
    values["all_feature_mean"] = (error * valid).sum() / (mask.sum() * 417)
    return total, values


class Trainer:
    def __init__(self, config, samples, identity, device="cuda:0"):
        self.config = config
        self.samples = samples
        self.identity = identity
        self.device = torch.device(device)
        self.model = make_model(config["model"]).to(self.device)
        self.ema = copy.deepcopy(self.model).eval().requires_grad_(False)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config["learning_rate"],
                                           weight_decay=0.0, foreach=False)
        self.diffusion = Diffusion(1000).to(self.device)
        self.batch_rng = torch.Generator().manual_seed(config["seed"] + 1)
        self.noise_rng = torch.Generator(device=self.device).manual_seed(config["seed"] + 2)
        self.step = 0
        self.ema_updates = 0
        self.best = float("inf")

    def update(self):
        self.model.train()
        indices = torch.randint(len(self.samples), (self.config["batch_size"],), generator=self.batch_rng).tolist()
        batch = batch_to([self.samples[i] for i in indices], self.device)
        t = torch.randint(1000, (len(indices),), generator=self.noise_rng, device=self.device)
        noise = torch.randn(batch["x0"].shape, generator=self.noise_rng, device=self.device)
        noisy = self.diffusion.q_sample(batch["x0"], t, noise)
        prediction = forward(self.model, noisy, batch, t)
        total, parts = losses(prediction, batch["x0"], batch["x_pad_mask"])
        if not torch.isfinite(total):
            raise FloatingPointError("Nonfinite training loss")
        self.optimizer.zero_grad(set_to_none=True)
        total.backward()
        norms = {}
        for name, stage in (("root", self.model.root_model), ("body", self.model.body_model)):
            grads = [p.grad for p in stage.parameters() if p.grad is not None]
            norm = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g) for g in grads]))
            if not torch.isfinite(norm) or norm <= 0:
                raise FloatingPointError(f"Invalid {name} gradients: {norm}")
            norms[name] = float(norm)
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config["grad_clip"],
                                                  error_if_nonfinite=True)
        self.optimizer.step()
        self.step += 1
        if self.step % self.config["ema_every"] == 0:
            self.ema_updates += 1
            # Warmup is explicit: short M3 runs must not average mostly random initialization.
            decay = min(self.config["ema_decay"], (1 + self.ema_updates) / (10 + self.ema_updates))
            with torch.no_grad():
                for a, p in zip(self.ema.parameters(), self.model.parameters()):
                    a.lerp_(p, 1 - decay)
        return {"loss": float(total.detach()), **{k: float(v.detach()) for k, v in parts.items()},
                "grad_norm": float(grad_norm), "root_grad_norm": norms["root"], "body_grad_norm": norms["body"]}

    @torch.inference_mode()
    def evaluate(self, samples, model=None):
        model = model or self.model
        was_training = model.training
        model.eval()
        batch = batch_to(samples, self.device)
        generator = torch.Generator(device=self.device).manual_seed(self.config["seed"] + 100)
        noise = torch.randn(batch["x0"].shape, generator=generator, device=self.device)
        results = {}
        for timestep in (0, 249, 499, 749, 999):
            t = torch.full((len(samples),), timestep, device=self.device, dtype=torch.long)
            pred = forward(model, self.diffusion.q_sample(batch["x0"], t, noise), batch, t)
            total, parts = losses(pred, batch["x0"], batch["x_pad_mask"])
            results[f"t{timestep:03d}"] = float(total)
            for k, v in parts.items():
                results[k] = results.get(k, 0.0) + float(v) / 5
        results["loss"] = sum(results[f"t{t:03d}"] for t in (0, 249, 499, 749, 999)) / 5
        if not all(np.isfinite(list(results.values()))):
            raise FloatingPointError("Nonfinite evaluation")
        model.train(was_training)
        return results

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {"schema": "kimodo-m3-checkpoint-v1", "config": self.config, "identity": self.identity,
                 "step": self.step, "ema_updates": self.ema_updates, "best": self.best,
                 "model": self.model.state_dict(), "ema": self.ema.state_dict(),
                 "optimizer": self.optimizer.state_dict(), "batch_rng": self.batch_rng.get_state(),
                 "noise_rng": self.noise_rng.get_state(), "torch_rng": torch.get_rng_state(),
                 "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                 "python_rng": random.getstate(), "numpy_rng": np.random.get_state()}
        tmp = path.with_suffix(".tmp")
        with tmp.open("wb") as handle:
            torch.save(state, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)

    def restore(self, path):
        # Only load checkpoints produced in this project, never arbitrary downloaded pickle files.
        state = torch.load(path, map_location="cpu", weights_only=False)
        if state["identity"] != self.identity or state["config"] != self.config:
            raise ValueError("Checkpoint data/subset/code/config mismatch")
        self.model.load_state_dict(state["model"])
        self.ema.load_state_dict(state["ema"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.step, self.ema_updates, self.best = state["step"], state["ema_updates"], state["best"]
        self.batch_rng.set_state(state["batch_rng"])
        self.noise_rng.set_state(state["noise_rng"])
        torch.set_rng_state(state["torch_rng"])
        if state["cuda_rng"]:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])


@torch.inference_mode()
def sample_ddim(model, batch, seed, steps=100):
    was_training = model.training
    model.eval()
    device = batch["x0"].device
    generator = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(batch["x0"].shape, device=device, generator=generator)
    diffusion = Diffusion(1000).to(device)  # Never mutate the training diffusion schedule.
    use_timesteps, mapping = diffusion.space_timesteps(steps)
    use_timesteps, mapping = use_timesteps[:steps], mapping[:steps]
    sampler = DDIMSampler(diffusion)
    for i in range(steps - 1, -1, -1):
        t = torch.full((len(x),), i, dtype=torch.long, device=device)
        predicted = forward(model, x, batch, mapping[t])
        x = sampler(use_timesteps, x, predicted, t)
        x = x.masked_fill(~batch["x_pad_mask"][..., None], 0)
    if not torch.isfinite(x).all():
        raise FloatingPointError("Nonfinite unconditional-noise DDIM output")
    model.train(was_training)
    return x


def motion_image(reference, generated, parents, title="", frames=5):
    """A shared-scale X/Y and Z/Y projection; blue=reference, orange=generated."""
    arrays = [np.asarray(reference), np.asarray(generated)]
    indices = np.linspace(0, len(reference) - 1, frames).astype(int)
    points = np.concatenate(arrays, axis=0).reshape(-1, 3)
    center = (points.min(0) + points.max(0)) / 2
    extent = max(float(np.ptp(points, axis=0).max()), 1.2) * 1.15
    width, cell, top = frames * 220, 220, 70
    canvas = Image.new("RGB", (width, top + cell * 2), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 8), title[:145], fill="black")
    draw.text((12, 30), "BLUE reference | ORANGE generated | raw, no postprocess | shared world scale", fill="black")
    for row, horizontal in enumerate((0, 2)):
        for col, t in enumerate(indices):
            x0, y0 = col * cell, top + row * cell
            draw.text((x0 + 8, y0 + 5), f"{'XY' if row == 0 else 'ZY'} frame {t}", fill="black")
            for positions, color in zip(arrays, ((32, 110, 210), (230, 95, 20))):
                coords = positions[t]
                px = (coords[:, horizontal] - center[horizontal]) / extent * 185 + x0 + 110
                py = -(coords[:, 1] - center[1]) / extent * 185 + y0 + 120
                for j, parent in enumerate(parents):
                    if parent >= 0:
                        draw.line((float(px[j]), float(py[j]), float(px[parent]), float(py[parent])), fill=color, width=2)
    return canvas


@torch.inference_mode()
def export_samples(model, samples, output, seed, steps=100, writer=None, step=0, tag="raw"):
    from kimodo.exports.mujoco import MujocoQposConverter
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device
    batch = batch_to(samples, device)
    prediction = sample_ddim(model, batch, seed, steps)
    converter = MujocoQposConverter(model.motion_rep.skeleton)
    parents = model.motion_rep.skeleton.joint_parents.cpu().tolist()
    report = []
    for i, sample in enumerate(samples):
        length = sample["length"]
        pred = prediction[i:i + 1, :length]
        ref = batch["x0"][i:i + 1, :length]
        decoded = model.motion_rep.inverse(pred, is_normalized=True)
        reference = model.motion_rep.inverse(ref, is_normalized=True)
        qpos = converter.dict_to_qpos(decoded, device=device)[0]
        if qpos.shape != (length, 36) or not np.isfinite(qpos).all():
            raise FloatingPointError("Invalid exported qpos")
        label = f"{i:02d}_{sample['group']}"
        np.savez(output / f"{label}.npz", **{k: v[0].cpu().numpy() for k, v in decoded.items()})
        np.savetxt(output / f"{label}.csv", qpos, delimiter=",")
        p, r = decoded["posed_joints"][0], reference["posed_joints"][0]
        error = torch.linalg.vector_norm(p - r, dim=-1).mean()
        relative = torch.linalg.vector_norm((p - p[:, :1]) - (r - r[:, :1]), dim=-1).mean()
        image = motion_image(r.cpu().numpy(), p.cpu().numpy(), parents, sample["text"])
        image.save(output / f"{label}.png")
        if writer:
            writer.add_image(f"samples/{tag}/{label}", np.array(image), step, dataformats="HWC")
        report.append({"sample_id": sample["sample_id"], "text": sample["text"], "group": sample["group"],
                       "frames": length, "world_mpjpe_m": float(error), "root_relative_mpjpe_m": float(relative),
                       "finite_qpos": True, "seed": seed, "first_heading_angle": float(sample["first_heading_angle"]),
                       "postprocess": False})
    atomic_json(output / "metrics.json", report)
    return {"world_mpjpe_m": float(np.mean([x["world_mpjpe_m"] for x in report])),
            "root_relative_mpjpe_m": float(np.mean([x["root_relative_mpjpe_m"] for x in report]))}
