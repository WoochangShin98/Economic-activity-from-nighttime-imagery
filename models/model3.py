# =========================
# 1. Imports
# =========================
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# =========================
# 2. Settings
# =========================
BACKGROUND_VALUE = 220.0
IMAGE_SIZE = (64, 64)
SEED = 5527


def set_seed(seed=5527):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)


# =========================
# 3. Load CSV
# =========================
final_df = pd.read_csv(
    r"C:\Users\tlsdn\OneDrive\Documents\바탕 화면\county_image_and_poverty_rate.csv",
    dtype={"county_id": str}
)

# Fix paths
final_df["image_path"] = final_df["image_path"].str.replace(
    r"C:\\Users\\zinsc\\OneDrive\\Desktop\\MS DataScience\\CSCI 5527 - Deep Learning\\Project\\Datasets\\opt1_county_only\\opt1_county_only",
    r"C:\\Users\\tlsdn\\OneDrive\\Documents\\바탕 화면\\opt1_county_only\\opt1_county_only",
    regex=True
)

final_df = final_df[["image_path", "poverty_rate"]].copy()

final_df["poverty_rate"] = pd.to_numeric(
    final_df["poverty_rate"],
    errors="coerce"
)

final_df = final_df.dropna(subset=["poverty_rate"])

final_df = final_df[
    final_df["image_path"].str.lower().str.endswith(".png")
].copy()

final_df = final_df[
    final_df["image_path"].apply(lambda x: Path(str(x)).exists())
].reset_index(drop=True)

print("Final dataframe shape:", final_df.shape)
print(final_df.head())
print(final_df["poverty_rate"].describe())


# =========================
# 4. Split
# =========================
train_df, temp_df = train_test_split(
    final_df,
    test_size=0.30,
    random_state=SEED
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=SEED
)

print("\nSplit sizes")
print("Train:", train_df.shape)
print("Val:  ", val_df.shape)
print("Test: ", test_df.shape)


# =========================
# 5. Load Image
# =========================
def load_gray_image(path, target_size=IMAGE_SIZE, background_value=BACKGROUND_VALUE):

    path = str(path)
    suffix = Path(path).suffix.lower()

    if suffix != ".png":
        raise ValueError(f"Expected PNG file, got {suffix}")

    pil_img = Image.open(path).convert("RGBA")

    arr = np.array(pil_img, dtype=np.float32)

    rgb = arr[..., :3]
    alpha = arr[..., 3] / 255.0

    gray = rgb.mean(axis=-1)

    # gray background composite
    img = alpha * gray + (1.0 - alpha) * background_value

    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)

    img = np.clip(img, a_min=0, a_max=None)

    # IMPORTANT
    img = np.log1p(img)

    img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    _, _, h, w = img_tensor.shape

    target_h, target_w = target_size

    scale = min(target_h / h, target_w / w)

    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))

    img_tensor = F.interpolate(
        img_tensor,
        size=(new_h, new_w),
        mode="bilinear",
        align_corners=False
    )

    pad_h = target_h - new_h
    pad_w = target_w - new_w

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top

    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    img_tensor = F.pad(
        img_tensor,
        (pad_left, pad_right, pad_top, pad_bottom),
        mode="constant",
        value=np.log1p(background_value)
    )

    img = img_tensor.squeeze(0).squeeze(0).numpy().astype(np.float32)

    return img


sample_img = load_gray_image(train_df.iloc[0]["image_path"])

print(sample_img.shape)
print(sample_img.min(), sample_img.max(), sample_img.mean())


# =========================
# 6. Handcrafted Features
# =========================
def extract_handcrafted_features(path, lit_threshold=0.10, bright_threshold=0.50):

    img = load_gray_image(path)

    pixels = img.reshape(-1)

    feats = {
        "mean_brightness": pixels.mean(),
        "max_brightness": pixels.max(),
        "std_brightness": pixels.std(),
        "median_brightness": np.median(pixels),
        "p25_brightness": np.percentile(pixels, 25),
        "p75_brightness": np.percentile(pixels, 75),
        "p90_brightness": np.percentile(pixels, 90),
        "p95_brightness": np.percentile(pixels, 95),

        "lit_pixel_share": np.mean(pixels > lit_threshold),
        "bright_pixel_share": np.mean(pixels > bright_threshold),

        "zero_pixel_share": np.mean(pixels <= 1e-6),
        "nonzero_pixel_share": np.mean(pixels > 1e-6)
    }

    return feats


def build_feature_df(df):

    feature_rows = []

    for _, row in df.iterrows():

        feats = extract_handcrafted_features(row["image_path"])

        feats["image_path"] = row["image_path"]
        feats["poverty_rate"] = row["poverty_rate"]

        feature_rows.append(feats)

    return pd.DataFrame(feature_rows)


train_feat_df = build_feature_df(train_df)
val_feat_df = build_feature_df(val_df)
test_feat_df = build_feature_df(test_df)

print(train_feat_df.head())


feature_cols = [
    "mean_brightness",
    "max_brightness",
    "std_brightness",
    "median_brightness",
    "p25_brightness",
    "p75_brightness",
    "p90_brightness",
    "p95_brightness",
    "lit_pixel_share",
    "bright_pixel_share",
    "zero_pixel_share",
    "nonzero_pixel_share"
]

X_train = train_feat_df[feature_cols].copy()
y_train = train_feat_df["poverty_rate"].copy()

X_val = val_feat_df[feature_cols].copy()
y_val = val_feat_df["poverty_rate"].copy()

X_test = test_feat_df[feature_cols].copy()
y_test = test_feat_df["poverty_rate"].copy()


# =========================
# 7. Metrics
# =========================
def regression_metrics(y_true, y_pred, name="Model"):

    mse = mean_squared_error(y_true, y_pred)

    rmse = np.sqrt(mse)

    mae = mean_absolute_error(y_true, y_pred)

    r2 = r2_score(y_true, y_pred)

    print(f"\n{name}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"R2:   {r2:.4f}")

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2
    }


# =========================
# 8. Ridge
# =========================
ridge_model = Pipeline([
    ("scaler", StandardScaler()),
    ("ridge", Ridge(alpha=1.0))
])

ridge_model.fit(X_train, y_train)

val_pred_ridge = ridge_model.predict(X_val)
test_pred_ridge = ridge_model.predict(X_test)

ridge_val_metrics = regression_metrics(
    y_val,
    val_pred_ridge,
    name="Ridge Validation"
)

ridge_test_metrics = regression_metrics(
    y_test,
    test_pred_ridge,
    name="Ridge Test"
)


# =========================
# 9. Random Forest
# =========================
rf_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=3,
    random_state=SEED,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

val_pred_rf = rf_model.predict(X_val)
test_pred_rf = rf_model.predict(X_test)

rf_val_metrics = regression_metrics(
    y_val,
    val_pred_rf,
    name="Random Forest Validation"
)

rf_test_metrics = regression_metrics(
    y_test,
    test_pred_rf,
    name="Random Forest Test"
)


# =========================
# 10. RF Importance
# =========================
rf_importance_df = pd.DataFrame({
    "feature": feature_cols,
    "importance": rf_model.feature_importances_
}).sort_values("importance", ascending=False)

print("\nRandom Forest feature importance")
print(rf_importance_df)


# =========================
# 11. CNN Dataset
# =========================
y_train_mean = train_df["poverty_rate"].mean()
y_train_std = train_df["poverty_rate"].std()

print("\ny_train_mean:", y_train_mean)
print("y_train_std:", y_train_std)


def compute_mean_std(df):

    total_sum = 0.0
    total_sq_sum = 0.0
    total_pixels = 0

    for path in df["image_path"]:

        img = load_gray_image(path)

        total_sum += img.sum()
        total_sq_sum += (img ** 2).sum()

        total_pixels += img.size

    mean = total_sum / total_pixels

    var = total_sq_sum / total_pixels - mean ** 2

    std = np.sqrt(max(var, 1e-8))

    return float(mean), float(std)


train_mean, train_std = compute_mean_std(train_df)

print("Train mean:", train_mean)
print("Train std:", train_std)


class CountyPovertyDataset1C(Dataset):

    def __init__(self, df, img_mean, img_std, y_mean, y_std):

        self.df = df.reset_index(drop=True).copy()

        self.img_mean = img_mean
        self.img_std = img_std

        self.y_mean = y_mean
        self.y_std = y_std

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        img = load_gray_image(row["image_path"])

        img = (img - self.img_mean) / (self.img_std + 1e-6)

        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)

        y = float(row["poverty_rate"])

        y = (y - self.y_mean) / (self.y_std + 1e-6)

        y = torch.tensor(y, dtype=torch.float32)

        return img, y


batch_size = 32

train_dataset = CountyPovertyDataset1C(
    train_df,
    train_mean,
    train_std,
    y_train_mean,
    y_train_std
)

val_dataset = CountyPovertyDataset1C(
    val_df,
    train_mean,
    train_std,
    y_train_mean,
    y_train_std
)

test_dataset = CountyPovertyDataset1C(
    test_df,
    train_mean,
    train_std,
    y_train_mean,
    y_train_std
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False
)


# =========================
# 12. CNN
# =========================
class CNNRegressor1C(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1,1))
        )

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, x, return_embedding=False):

        x = self.features(x)

        embedding = torch.flatten(x, start_dim=1)

        out = self.regressor(x).squeeze(1)

        if return_embedding:
            return out, embedding

        return out


# =========================
# 13. Train CNN
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("\nUsing device:", device)

model = CNNRegressor1C().to(device)

criterion = nn.SmoothL1Loss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=3e-4,
    weight_decay=1e-4
)


def evaluate_regression(model, loader, criterion, device, y_mean, y_std):

    model.eval()

    all_preds_std = []
    all_targets_std = []

    running_loss = 0.0

    with torch.no_grad():

        for images, targets in loader:

            images = images.to(device)
            targets = targets.to(device)

            preds = model(images)

            loss = criterion(preds, targets)

            running_loss += loss.item() * images.size(0)

            all_preds_std.extend(preds.cpu().numpy())
            all_targets_std.extend(targets.cpu().numpy())

    all_preds_std = np.array(all_preds_std)
    all_targets_std = np.array(all_targets_std)

    all_preds = all_preds_std * (y_std + 1e-6) + y_mean
    all_targets = all_targets_std * (y_std + 1e-6) + y_mean

    mse = mean_squared_error(all_targets, all_preds)

    rmse = np.sqrt(mse)

    mae = mean_absolute_error(all_targets, all_preds)

    r2 = r2_score(all_targets, all_preds)

    avg_loss = running_loss / len(loader.dataset)

    return {
        "loss": avg_loss,
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2
    }


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    y_mean,
    y_std,
    epochs=50,
    patience = 15
):

    best_val_loss = float("inf")

    best_state = None

    patience_counter = 0

    for epoch in range(1, epochs + 1):

        model.train()

        running_loss = 0.0

        for images, targets in train_loader:

            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            preds = model(images)

            loss = criterion(preds, targets)

            loss.backward()

            optimizer.step()

            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(train_loader.dataset)

        val_metrics = evaluate_regression(
            model,
            val_loader,
            criterion,
            device,
            y_mean,
            y_std
        )

        print(
            f"Epoch {epoch:02d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val RMSE: {val_metrics['rmse']:.4f} | "
            f"Val MAE: {val_metrics['mae']:.4f} | "
            f"Val R2: {val_metrics['r2']:.4f}"
        )

        if val_metrics["loss"] < best_val_loss:

            best_val_loss = val_metrics["loss"]

            best_state = model.state_dict()

            patience_counter = 0

        else:

            patience_counter += 1

        if patience_counter >= patience:

            print("Early stopping triggered.")

            break

    if best_state is not None:

        model.load_state_dict(best_state)

    return model


model = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    device=device,
    y_mean=y_train_mean,
    y_std=y_train_std
)


test_metrics_cnn = evaluate_regression(
    model,
    test_loader,
    criterion,
    device,
    y_mean=y_train_mean,
    y_std=y_train_std
)

print("\nEnd-to-end CNN Test")
print(f"RMSE: {test_metrics_cnn['rmse']:.4f}")
print(f"MAE:  {test_metrics_cnn['mae']:.4f}")
print(f"R2:   {test_metrics_cnn['r2']:.4f}")


# =========================
# 14. Embeddings
# =========================
def extract_embeddings(model, loader, device, y_mean, y_std):

    model.eval()

    all_embeddings = []

    all_targets_std = []
    all_preds_std = []

    with torch.no_grad():

        for images, targets in loader:

            images = images.to(device)

            preds, embeddings = model(images, return_embedding=True)

            all_embeddings.append(embeddings.cpu().numpy())

            all_targets_std.extend(targets.numpy())
            all_preds_std.extend(preds.cpu().numpy())

    X = np.vstack(all_embeddings)

    y_std_arr = np.array(all_targets_std)

    y = y_std_arr * (y_std + 1e-6) + y_mean

    cnn_preds = np.array(all_preds_std) * (y_std + 1e-6) + y_mean

    return X, y, cnn_preds


X_train_emb, y_train_emb, train_cnn_pred = extract_embeddings(
    model,
    train_loader,
    device,
    y_train_mean,
    y_train_std
)

X_val_emb, y_val_emb, val_cnn_pred = extract_embeddings(
    model,
    val_loader,
    device,
    y_train_mean,
    y_train_std
)

X_test_emb, y_test_emb, test_cnn_pred = extract_embeddings(
    model,
    test_loader,
    device,
    y_train_mean,
    y_train_std
)

print("Embedding train shape:", X_train_emb.shape)
print("Embedding val shape:  ", X_val_emb.shape)
print("Embedding test shape: ", X_test_emb.shape)

# =========================
# 15. Ridge on CNN Embeddings
# =========================
ridge_emb_model = Pipeline([
    ("scaler", StandardScaler()),
    ("ridge", Ridge(alpha=1.0))
])

ridge_emb_model.fit(X_train_emb, y_train_emb)

val_pred_ridge_emb = ridge_emb_model.predict(X_val_emb)
test_pred_ridge_emb = ridge_emb_model.predict(X_test_emb)

ridge_emb_val_metrics = regression_metrics(
    y_val_emb,
    val_pred_ridge_emb,
    name="Ridge on CNN Embeddings Validation"
)

ridge_emb_test_metrics = regression_metrics(
    y_test_emb,
    test_pred_ridge_emb,
    name="Ridge on CNN Embeddings Test"
)

# =========================
# 16. Random Forest on CNN Embeddings
# =========================
rf_emb_model = RandomForestRegressor(
    n_estimators=500,
    max_depth=10,
    min_samples_leaf=3,
    random_state=SEED,
    n_jobs=-1
)

rf_emb_model.fit(X_train_emb, y_train_emb)

val_pred_rf_emb = rf_emb_model.predict(X_val_emb)
test_pred_rf_emb = rf_emb_model.predict(X_test_emb)

rf_emb_val_metrics = regression_metrics(
    y_val_emb,
    val_pred_rf_emb,
    name="RF on CNN Embeddings Validation"
)

rf_emb_test_metrics = regression_metrics(
    y_test_emb,
    test_pred_rf_emb,
    name="RF on CNN Embeddings Test"
)

# =========================
# 17. Final Comparison
# =========================
hybrid_results_df = pd.DataFrame([
    {
        "model": "Ridge_handcrafted",
        "test_rmse": ridge_test_metrics["rmse"],
        "test_mae": ridge_test_metrics["mae"],
        "test_r2": ridge_test_metrics["r2"],
    },
    {
        "model": "RF_handcrafted",
        "test_rmse": rf_test_metrics["rmse"],
        "test_mae": rf_test_metrics["mae"],
        "test_r2": rf_test_metrics["r2"],
    },
    {
        "model": "End_to_end_CNN",
        "test_rmse": test_metrics_cnn["rmse"],
        "test_mae": test_metrics_cnn["mae"],
        "test_r2": test_metrics_cnn["r2"],
    },
    {
        "model": "Ridge_on_CNN_Embeddings",
        "test_rmse": ridge_emb_test_metrics["rmse"],
        "test_mae": ridge_emb_test_metrics["mae"],
        "test_r2": ridge_emb_test_metrics["r2"],
    },
    {
        "model": "RF_on_CNN_Embeddings",
        "test_rmse": rf_emb_test_metrics["rmse"],
        "test_mae": rf_emb_test_metrics["mae"],
        "test_r2": rf_emb_test_metrics["r2"],
    }
])

print("\nFinal model comparison")
print(hybrid_results_df)

hybrid_results_df.to_csv("hybrid_results_improved.csv", index=False)
rf_importance_df.to_csv("rf_feature_importance_improved.csv", index=False)

print("\nSaved hybrid_results_improved.csv")
print("Saved rf_feature_importance_improved.csv")