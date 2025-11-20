"""Three-stage training pipeline for SD-MKD.

This module orchestrates:
1) Teacher pretraining on dataset A.
2) Stacking ensemble training on dataset B.
3) Student distillation on dataset B using CE + FKL + RKL + Sinkhorn losses.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from data_preprocessing import quick_load_dataset
from .loss_functions import class_cost_matrix, distillation_loss
from .monitor import LiveTrainingMonitor
from models import (
    StudentNet,
    DenseNet121Teacher,
    MobileNetV3LargeTeacher,
    ResNet50Teacher,
    StackingModel,
    TeacherEnsemble,
    train_single_teacher,
    train_stacking_model,
)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _validate_classifier(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total = 0
    correct = 0
    losses = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            preds = logits.argmax(dim=-1)
            total += y.numel()
            correct += (preds == y).sum().item()
            losses.append(loss.item())
    val_loss = float(sum(losses) / max(1, len(losses)))
    val_acc = correct / max(1, total)
    return val_loss, val_acc


# ---------------------------------------------------------------------------
# Stage I: teacher pretraining
# ---------------------------------------------------------------------------


def train_teachers(
    train_loader_A: DataLoader,
    val_loader_A: DataLoader,
    num_classes: int,
    device: torch.device,
    num_epochs_teacher: int = 1,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    monitor: LiveTrainingMonitor | None = None,
) -> Dict[str, str]:
    """Train three teachers independently and save checkpoints.

    Returns a mapping from teacher name to checkpoint path for convenience.
    """

    teachers = {
        "resnet50": ResNet50Teacher(num_classes, pretrained=False).to(device),
        "mbv3": MobileNetV3LargeTeacher(num_classes, pretrained=False).to(device),
        "densenet121": DenseNet121Teacher(num_classes, pretrained=False).to(device),
    }
    ckpt_paths: Dict[str, str] = {}
    for name, model in teachers.items():
        stage_name = f"teacher_{name}"
        if monitor:
            monitor.stage_start(stage_name, num_epochs_teacher)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        for res in train_single_teacher(
            model, train_loader_A, val_loader_A, device, epochs=num_epochs_teacher, lr=lr, weight_decay=weight_decay
        ):
            print(
                f"[Teacher {name}] epoch={res.epoch} train_loss={res.train_loss:.4f} "
                f"val_loss={res.val_loss:.4f} val_acc={res.val_acc:.4f}"
            )
            if monitor:
                monitor.log(
                    stage_name,
                    res.epoch,
                    {"train_loss": res.train_loss, "val_loss": res.val_loss, "val_acc": res.val_acc},
                )
        path = f"{name}_teacher.pth"
        torch.save(model.state_dict(), path)
        ckpt_paths[name] = path
        if monitor:
            monitor.stage_end(stage_name)
    return ckpt_paths


# ---------------------------------------------------------------------------
# Stage II: stacking ensemble
# ---------------------------------------------------------------------------


def train_stacking_model_stage(
    train_loader_B: DataLoader,
    val_loader_B: DataLoader,
    num_classes: int,
    device: torch.device,
    num_epochs_stacking: int = 1,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    teacher_ckpts: Dict[str, str] | None = None,
    monitor: LiveTrainingMonitor | None = None,
) -> str:
    """Train the stacking MLP on top of frozen teachers.

    Returns the stacking checkpoint path.
    """

    # Load teachers
    t1 = ResNet50Teacher(num_classes, pretrained=False)
    t2 = MobileNetV3LargeTeacher(num_classes, pretrained=False)
    t3 = DenseNet121Teacher(num_classes, pretrained=False)
    if teacher_ckpts:
        t1.load_state_dict(torch.load(teacher_ckpts["resnet50"], map_location="cpu"))
        t2.load_state_dict(torch.load(teacher_ckpts["mbv3"], map_location="cpu"))
        t3.load_state_dict(torch.load(teacher_ckpts["densenet121"], map_location="cpu"))

    for t in (t1, t2, t3):
        t.to(device).eval()
        for p in t.parameters():
            p.requires_grad = False

    ensemble = TeacherEnsemble(num_classes=num_classes)
    ensemble.t1 = t1
    ensemble.t2 = t2
    ensemble.t3 = t3

    stacking = StackingModel(num_classes=num_classes).to(device)
    if monitor:
        monitor.stage_start("stacking", num_epochs_stacking)

    for res in train_stacking_model(
        ensemble,
        stacking,
        train_loader_B,
        val_loader_B,
        device,
        epochs=num_epochs_stacking,
        lr=lr,
        weight_decay=weight_decay,
    ):
        print(
            f"[Stacking] epoch={res.epoch} train_loss={res.train_loss:.4f} "
            f"val_loss={res.val_loss:.4f} val_acc={res.val_acc:.4f}"
        )
        if monitor:
            monitor.log(
                "stacking",
                res.epoch,
                {"train_loss": res.train_loss, "val_loss": res.val_loss, "val_acc": res.val_acc},
            )

    stacking_path = "stacking_model.pth"
    torch.save(stacking.state_dict(), stacking_path)
    if monitor:
        monitor.stage_end("stacking")
    return stacking_path


# ---------------------------------------------------------------------------
# Stage III: student distillation
# ---------------------------------------------------------------------------


def train_student_stage(
    train_loader_B: DataLoader,
    val_loader_B: DataLoader,
    num_classes: int,
    device: torch.device,
    T: float = 4.0,
    lamb_ce: float = 1.0,
    lamb_f: float = 0.5,
    lamb_r: float = 0.5,
    lamb_s: float = 0.1,
    num_epochs_student: int = 1,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    grad_clip: float | None = 5.0,
    scheduler_patience: int = 2,
    scheduler_factor: float = 0.5,
    min_lr: float = 1e-5,
    teacher_ckpts: Dict[str, str] | None = None,
    stacking_ckpt: str | None = None,
    monitor: LiveTrainingMonitor | None = None,
) -> str:
    """Distill the student from frozen teachers + stacking ensemble."""

    # Load frozen teachers and stacking
    t1 = ResNet50Teacher(num_classes, pretrained=False)
    t2 = MobileNetV3LargeTeacher(num_classes, pretrained=False)
    t3 = DenseNet121Teacher(num_classes, pretrained=False)
    stacking = StackingModel(num_classes=num_classes)

    if teacher_ckpts:
        t1.load_state_dict(torch.load(teacher_ckpts["resnet50"], map_location="cpu"))
        t2.load_state_dict(torch.load(teacher_ckpts["mbv3"], map_location="cpu"))
        t3.load_state_dict(torch.load(teacher_ckpts["densenet121"], map_location="cpu"))
    if stacking_ckpt:
        stacking.load_state_dict(torch.load(stacking_ckpt, map_location="cpu"))

    for m in (t1, t2, t3, stacking):
        m.to(device).eval()
        for p in m.parameters():
            p.requires_grad = False

    student = StudentNet(num_classes=num_classes).to(device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=scheduler_factor, patience=scheduler_patience, min_lr=min_lr
    )
    cost = class_cost_matrix(num_classes, device=device)

    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    if monitor:
        monitor.stage_start("student", num_epochs_student)

    for epoch in range(1, num_epochs_student + 1):
        student.train()
        losses = []
        for x, y in train_loader_B:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                l1 = t1(x)
                l2 = t2(x)
                l3 = t3(x)
                teacher_logits = stacking(l1, l2, l3)
            with torch.cuda.amp.autocast(enabled=use_amp):
                student_logits = student(x)
                loss = distillation_loss(
                    student_logits=student_logits,
                    teacher_logits=teacher_logits,
                    labels=y,
                    T=T,
                    lamb_ce=lamb_ce,
                    lamb_f=lamb_f,
                    lamb_r=lamb_r,
                    lamb_s=lamb_s,
                    cost_matrix=cost,
                )
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            if grad_clip and grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            losses.append(loss.item())
        train_loss = float(sum(losses) / max(1, len(losses)))
        val_loss, val_acc = _validate_classifier(student, val_loader_B, device)
        scheduler.step(val_loss)
        print(
            f"[Student] epoch={epoch} train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )
        if monitor:
            monitor.log("student", epoch, {"train_loss": train_loss, "val_loss": val_loss, "val_acc": val_acc})

    student_path = "student_sd_mkd.pth"
    torch.save(student.state_dict(), student_path)
    if monitor:
        monitor.stage_end("student")
    return student_path


# ---------------------------------------------------------------------------
# CLI entry (also reusable by main.py)
# ---------------------------------------------------------------------------


def _make_dummy_loader(num_samples: int, num_classes: int, height: int, width: int, batch_size: int) -> DataLoader:
    x = torch.rand(num_samples, 1, height, width)
    y = torch.randint(0, num_classes, (num_samples,))
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True)


def run_demo_pipeline(mode: str, device: torch.device, monitor: LiveTrainingMonitor | None = None):
    num_classes = 5
    h = w = 32
    batch_size = 8

    # Dummy loaders for demonstration
    train_A = _make_dummy_loader(32, num_classes, h, w, batch_size)
    val_A = _make_dummy_loader(16, num_classes, h, w, batch_size)
    train_B = _make_dummy_loader(32, num_classes, h, w, batch_size)
    val_B = _make_dummy_loader(16, num_classes, h, w, batch_size)

    teacher_ckpts: Dict[str, str] | None = None
    stacking_ckpt: str | None = None

    def _ensure_teachers() -> Dict[str, str]:
        paths = {
            "resnet50": "resnet50_teacher.pth",
            "mbv3": "mbv3_teacher.pth",
            "densenet121": "densenet121_teacher.pth",
        }
        if all(os.path.exists(p) for p in paths.values()):
            return paths
        print("[Demo] Training teachers because checkpoints were not found.")
        return train_teachers(train_A, val_A, num_classes, device, monitor=monitor)

    def _ensure_stacking(existing_teachers: Dict[str, str]) -> str:
        path = "stacking_model.pth"
        if os.path.exists(path):
            return path
        print("[Demo] Training stacking model because checkpoint was not found.")
        return train_stacking_model_stage(
            train_B,
            val_B,
            num_classes,
            device,
            teacher_ckpts=existing_teachers,
            monitor=monitor,
        )

    if mode == "train_teachers":
        train_teachers(train_A, val_A, num_classes, device, monitor=monitor)
    elif mode == "train_stacking":
        teacher_ckpts = _ensure_teachers()
        train_stacking_model_stage(
            train_B,
            val_B,
            num_classes,
            device,
            teacher_ckpts=teacher_ckpts,
            monitor=monitor,
        )
    elif mode == "train_student":
        teacher_ckpts = _ensure_teachers()
        stacking_ckpt = _ensure_stacking(teacher_ckpts)
        train_student_stage(
            train_B,
            val_B,
            num_classes,
            device,
            teacher_ckpts=teacher_ckpts,
            stacking_ckpt=stacking_ckpt,
            monitor=monitor,
        )
    else:
        raise ValueError(f"Unsupported mode: {mode}")


def run_real_pipeline(
    mode: str,
    device: torch.device,
    dataset_name: str,
    data_root: str,
    batch_size: int = 64,
    num_workers: int = 4,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    max_samples_per_class: int | None = None,
    num_epochs_teacher: int = 5,
    num_epochs_stacking: int = 5,
    num_epochs_student: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    student_grad_clip: float | None = 5.0,
    student_min_lr: float = 1e-5,
    student_lr_patience: int = 2,
    student_lr_factor: float = 0.5,
    T: float = 4.0,
    lamb_ce: float = 1.0,
    lamb_f: float = 0.5,
    lamb_r: float = 0.5,
    lamb_s: float = 0.1,
    output_dir: str = "./checkpoints",
    monitor: LiveTrainingMonitor | None = None,
) -> Dict[str, str]:
    """Run teacher/stacking/student stages on real preprocessed image data."""

    # Load dataset
    train_loader, val_loader, test_loader, meta = quick_load_dataset(
        dataset_name=dataset_name,
        dataset_root=data_root,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        max_samples_per_class=max_samples_per_class,
    )
    num_classes = meta["num_classes"]
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def _ckpt(name: str) -> Path:
        return out_dir / name

    def _ensure_teachers() -> Dict[str, str]:
        paths = {
            "resnet50": str(_ckpt("resnet50_teacher.pth")),
            "mbv3": str(_ckpt("mbv3_teacher.pth")),
            "densenet121": str(_ckpt("densenet121_teacher.pth")),
        }
        if all(Path(p).exists() for p in paths.values()):
            return paths
        ckpts = train_teachers(
            train_loader,
            val_loader,
            num_classes,
            device,
            num_epochs_teacher=num_epochs_teacher,
            lr=lr,
            weight_decay=weight_decay,
            monitor=monitor,
        )
        # Move to output dir
        for k, src in ckpts.items():
            target = _ckpt(Path(src).name)
            Path(src).replace(target)
            paths[k] = str(target)
        return paths

    def _ensure_stacking(teachers: Dict[str, str]) -> str:
        path = str(_ckpt("stacking_model.pth"))
        if Path(path).exists():
            return path
        ckpt = train_stacking_model_stage(
            train_loader,
            val_loader,
            num_classes,
            device,
            teacher_ckpts=teachers,
            num_epochs_stacking=num_epochs_stacking,
            lr=lr,
            weight_decay=weight_decay,
            monitor=monitor,
        )
        Path(ckpt).replace(path)
        return path

    results: Dict[str, str] = {}

    if mode == "train_teachers":
        results = _ensure_teachers()
    elif mode == "train_stacking":
        teachers = _ensure_teachers()
        stack_path = _ensure_stacking(teachers)
        results = {"stacking": stack_path, **teachers}
    elif mode == "train_student":
        teachers = _ensure_teachers()
        stack_path = _ensure_stacking(teachers)
        student_ckpt = train_student_stage(
            train_loader,
            val_loader,
            num_classes,
            device,
            T=T,
            lamb_ce=lamb_ce,
            lamb_f=lamb_f,
            lamb_r=lamb_r,
            lamb_s=lamb_s,
            num_epochs_student=num_epochs_student,
            lr=lr,
            weight_decay=weight_decay,
            teacher_ckpts=teachers,
            stacking_ckpt=stack_path,
            monitor=monitor,
            grad_clip=student_grad_clip,
            scheduler_patience=student_lr_patience,
            scheduler_factor=student_lr_factor,
            min_lr=student_min_lr,
        )
        target = _ckpt(Path(student_ckpt).name)
        Path(student_ckpt).replace(target)
        results = {
            "student": str(target),
            "stacking": stack_path,
            **teachers,
        }
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    return results


def main():
    parser = argparse.ArgumentParser(description="SD-MKD three-stage training")
    parser.add_argument(
        "--mode",
        type=str,
        default="train_student",
        choices=["train_teachers", "train_stacking", "train_student"],
        help="Which training phase to run",
    )
    parser.add_argument("--use_real_data", action="store_true", help="Use real dataset instead of demo random data")
    parser.add_argument("--dataset", type=str, default="ISCXVPN2016", help="Dataset name (folder under dataset_root)")
    parser.add_argument("--dataset_root", type=str, default="/walnut_data/yqm/Dataset", help="Dataset root path")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--max_samples_per_class", type=int, default=0, help="0 for all samples")
    parser.add_argument("--epochs_teacher", type=int, default=5)
    parser.add_argument("--epochs_stacking", type=int, default=5)
    parser.add_argument("--epochs_student", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--student_grad_clip", type=float, default=5.0)
    parser.add_argument("--student_min_lr", type=float, default=1e-5)
    parser.add_argument("--student_lr_patience", type=int, default=2)
    parser.add_argument("--student_lr_factor", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--lamb_ce", type=float, default=1.0)
    parser.add_argument("--lamb_f", type=float, default=0.5)
    parser.add_argument("--lamb_r", type=float, default=0.5)
    parser.add_argument("--lamb_s", type=float, default=0.1)
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    monitor = LiveTrainingMonitor()
    if args.use_real_data:
        run_real_pipeline(
            mode=args.mode,
            device=device,
            dataset_name=args.dataset,
            data_root=args.dataset_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            max_samples_per_class=None if args.max_samples_per_class <= 0 else args.max_samples_per_class,
            num_epochs_teacher=args.epochs_teacher,
            num_epochs_stacking=args.epochs_stacking,
            num_epochs_student=args.epochs_student,
            lr=args.lr,
            weight_decay=args.weight_decay,
            student_grad_clip=args.student_grad_clip,
            student_min_lr=args.student_min_lr,
            student_lr_patience=args.student_lr_patience,
            student_lr_factor=args.student_lr_factor,
            T=args.temperature,
            lamb_ce=args.lamb_ce,
            lamb_f=args.lamb_f,
            lamb_r=args.lamb_r,
            lamb_s=args.lamb_s,
            output_dir=args.output_dir,
            monitor=monitor,
        )
    else:
        run_demo_pipeline(args.mode, device, monitor=monitor)


if __name__ == "__main__":
    main()
