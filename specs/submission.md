# SPEC: submit()

## PURPOSE

Runs inference on all test images using a trained `MultiTaskCelebNet` and writes a submission CSV to disk. The output mirrors the `train_data.csv` column schema so it can be submitted directly to the Kaggle competition.

---

## INPUTS

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `dataloader_test` | `DataLoader` | — | Test data. Each batch is `(imgs, ids)` where `imgs` is `Tensor[B, C, H, W]` and `ids` is a `list[str]` of B image IDs (filename without `.jpg`). |
| `model` | `MultiTaskCelebNet` | — | Trained model. `forward(imgs)` must return `(bin_logits, cls_logits)` with shapes `(B, 8)` and `(B, 501)`. |
| `columns_name` | `list[str]` | — | Ordered list of 10 column names: `["id", "No_Beard", "Young", "Mouth_Slightly_Open", "Smiling", "Male", "Wavy_Hair", "Black_Hair", "Wearing_Hat", "celebrity_id"]`. Drives the DataFrame schema. |
| `output_path` | `Path` | `Path("submissions/submission.csv")` | Destination path for the output CSV. Parent directory must already exist. |

### DataLoader contract

`dataloader_test` must yield `(imgs, ids)` per batch:
- `imgs`: `Tensor` of shape `(B, C, H, W)`, float32, values in [0, 1]
- `ids`: `list[str]` of length B — the image stem (filename without `.jpg`)

This requires a `CelebTestDataset` (no CSV, no label column) whose `__getitem__` returns `(image_tensor, stem)`. The DataLoader's default collation produces the correct `(Tensor, list[str])` batch format automatically.

### External globals

| Name | Value | Role |
|---|---|---|
| `device` | `torch.device` | All input tensors are moved here before the forward pass |

---

## OUTPUTS

Returns `None`. Side effect: writes a CSV file to `output_path`.

### CSV format

Mirrors `train_data.csv` column schema exactly:

| Column | Dtype | Values |
|---|---|---|
| `id` | String | image stem (e.g. `"20584"`) |
| `No_Beard` … `Wearing_Hat` | Int8 | 0 or 1 (thresholded at sigmoid ≥ 0.5) |
| `celebrity_id` | Int64 | 0–500 (argmax of class logits) |

One row per test image. Row order matches the iteration order of `dataloader_test`.

---

## ALGORITHM

```python
def submit(dataloader_test, model, columns_name,
           output_path=Path("submissions/submission.csv")):

    model.eval()
    all_ids, all_bin, all_cls = [], [], []

    with torch.no_grad():
        for imgs, ids in dataloader_test:
            imgs = imgs.to(device)
            bin_logits, cls_logits = model(imgs)

            bin_preds = (torch.sigmoid(bin_logits) >= 0.5).int().cpu()  # (B, 8)
            cls_preds = cls_logits.argmax(1).cpu()                       # (B,)

            all_ids.extend(ids)
            all_bin.append(bin_preds)
            all_cls.append(cls_preds)

    bin_tensor = torch.cat(all_bin)   # (N, 8)
    cls_tensor = torch.cat(all_cls)   # (N,)

    df = pl.DataFrame(
        {
            columns_name[0]: all_ids,
            **{columns_name[i + 1]: bin_tensor[:, i].numpy() for i in range(8)},
            columns_name[9]: cls_tensor.numpy(),
        },
        schema={
            columns_name[0]: pl.String,
            **{columns_name[i + 1]: pl.Int8 for i in range(8)},
            columns_name[9]: pl.Int64,
        },
    )

    df.write_csv(output_path)
    print(f"Submission saved: {output_path} ({len(df)} rows)")
```

---

## INVARIANTS

- `model.eval()` is called before any forward pass; gradients are disabled throughout via `torch.no_grad()`.
- `columns_name` must have exactly 10 elements: index 0 = id, indices 1–8 = binary attributes in label order, index 9 = `celebrity_id`.
- Output row count equals the total number of samples in `dataloader_test`.
- Binary predictions are integers (0 or 1); `celebrity_id` predictions are integers (0–500).
- The parent directory of `output_path` must already exist; the function does not create it.
- The function does not modify model weights or optimizer state.
