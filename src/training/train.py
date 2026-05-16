"""Train SRCNN or dual-branch SR model on PROBA-V (ISRO PS-12)."""

import argparse
import os

from tensorflow import keras
from tensorflow.keras.callbacks import (
    CSVLogger,
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)

from src.data.data_generator import PROBAVDataGenerator
from src.data.dataset_loader import prepare_dataset
from src.models.srcnn import build_srcnn, psnr_metric
from src.training.losses import perceptual_mse_loss

LEARNING_RATE = 1e-4

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WEIGHTS_DIR = os.path.join(PROJECT_ROOT, "weights")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
TRAINING_LOG_PATH = os.path.join(RESULTS_DIR, "training_log.csv")

MODEL_WEIGHTS = {
    "srcnn": os.path.join(WEIGHTS_DIR, "best_model.h5"),
    "dual": os.path.join(WEIGHTS_DIR, "best_dual_model.h5"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train dual-input SR model on PROBA-V.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for train/val.")
    parser.add_argument(
        "--model",
        type=str,
        choices=["srcnn", "dual"],
        default="srcnn",
        help="Model architecture: srcnn (baseline) or dual (shared-weight branches).",
    )
    return parser.parse_args()


def ensure_dirs() -> None:
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)


def main() -> None:
    args = parse_args()
    ensure_dirs()

    train_scenes, val_scenes, test_scenes = prepare_dataset()
    print(
        f"Scenes — train: {len(train_scenes)}, val: {len(val_scenes)}, test: {len(test_scenes)}"
    )

    train_gen = PROBAVDataGenerator(
        train_scenes,
        batch_size=args.batch_size,
        augment=True,
    )
    val_gen = PROBAVDataGenerator(
        val_scenes,
        batch_size=args.batch_size,
        augment=False,
    )

    best_model_path = MODEL_WEIGHTS[args.model]
    print(f"Training model: {args.model} → {best_model_path}")

    if args.model == "dual":
        from src.models.dual_sr_model import build_dual_sr

        model = build_dual_sr(input_shape=(384, 384, 2))
    else:
        model = build_srcnn(input_shape=(384, 384, 2))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=perceptual_mse_loss,
        metrics=[psnr_metric],
    )

    callbacks = [
        ModelCheckpoint(
            best_model_path,
            monitor="val_loss",
            save_best_only=True,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
        ),
        CSVLogger(TRAINING_LOG_PATH),
    ]

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    val_loss, val_psnr = model.evaluate(val_gen, verbose=0)
    print(f"Final validation — loss: {val_loss:.6f}, PSNR: {val_psnr:.4f} dB")

    if "val_psnr_metric" in history.history:
        last_logged = history.history["val_psnr_metric"][-1]
        print(f"Last epoch logged val_psnr_metric: {last_logged:.4f} dB")


if __name__ == "__main__":
    main()
