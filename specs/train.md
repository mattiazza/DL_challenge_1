# SPEC: train()

## PURPOSE

Multi-task training loop for `MultiTaskCelebNet`. Jointly minimizes two losses per epoch:
- `BCEWithLogitsLoss` over 8 binary facial attributes (No_Beard, Young, Mouth_Slightly_Open, Smiling, Male, Wavy_Hair, Black_Hair, Wearing_Hat)
- `CrossEntropyLoss` over 501 celebrity identity classes (500 celebrities + "other")

The combined loss applies an optional per-epoch scalar weight to the classification head: `loss = binary_loss + w * cls_loss`, where `w = cls_weight_fn(epoch, prev_bin_loss, prev_cls_loss)` if provided, else `w = 1.0`. Tracks training and validation metrics (loss, binary accuracy, classification accuracy) across all epochs.

Supports two output modes:
- **Default** (`hparam_tuning=False`): live hiddenlayer plots updated each epoch (3 plots: loss, binary acc, classification acc — each showing train vs. val)
- **Tuning** (`hparam_tuning=True`): prints one console line per epoch, no plots

---

## INPUTS

### Parameters

| Parameter | Type | Description |
|---|---|---|
| `model` | `MultiTaskCelebNet` | The network to train. `forward(img)` must return `(bin_logits, cls_logits)` with shapes `(B, 8)` and `(B, 501)`. |
| `optimizer` | `torch.optim.Optimizer` | Pre-constructed Adam optimizer wrapping `model.parameters()`, lr=1e-3. Passed in, not created inside this function. |
| `dataloader_train` | `torch.utils.data.DataLoader` | Training data. Each batch is `(img, label)` where `img` shape is `(B, C, H, W)` and `label` shape is `(B, 9)`. |
| `dataloader_val` | `torch.utils.data.DataLoader` | Validation data. Same format as `dataloader_train`. |
| `epochs` | `int` | Number of full passes over the training data. |
| `hparam_tuning` | `bool` (default `False`) | Controls logging mode. `False` → hiddenlayer visualization; `True` → console print. |
| `cls_weight_fn` | `(int, float, float) -> float \| None` (default `None`) | If provided, called as `cls_weight_fn(epoch, prev_bin_loss, prev_cls_loss)` at the start of each epoch. `prev_bin_loss` and `prev_cls_loss` are the unweighted per-batch average training losses from the previous epoch (`binary_criterion` and `class_criterion` separately, **not** scaled by `cls_w`); both are `0.0` at epoch 0. The callable must handle `epoch == 0` (no meaningful previous losses). Returns the scalar weight `w` applied to `class_criterion` in the combined loss. If `None`, weight is `1.0`. Example: `lambda epoch, b, c: 1/8 if epoch == 0 else b/c`. |
| `early_stopping_patience` | `int \| None` (default `None`) | If `None`, no early stopping — training runs for all `epochs`. If an `int`, stops after this many consecutive epochs with no decrease in validation loss. Best model weights are restored before the function returns. |
| `binary_criterion` | `nn.BCEWithLogitsLoss` | Loss function for the 8 binary outputs. |
| `class_criterion` | `nn.CrossEntropyLoss` | Loss function for the 501-class output. |

### Label tensor layout
`label` has shape `(B, 9)`:
- `label[:, 0:8]` — float32 binary attribute labels (0.0 or 1.0), one column per attribute in the order above
- `label[:, 8]` — float32 celebrity ID (must be cast to `.long()` before use with CrossEntropyLoss)

### External globals
These must exist in the enclosing scope:

| Name | Value | Role |
|---|---|---|
| `device` | `torch.device` | All tensors are moved here before computation |


---

## OUTPUTS

Returns a 6-tuple of Python lists, each of length equal to the number of epochs actually run (≤ `epochs`):

```python
(loss_train, bin_acc_train, cls_acc_train, loss_val, bin_acc_val, cls_acc_val)
```

| Return value | Type | Range | Description |
|---|---|---|---|
| `loss_train` | `List[float]` | ≥ 0 | Combined loss averaged over training batches (denominator: `n_train_batches`) |
| `bin_acc_train` | `List[float]` | [0, 1] | Mean per-sample binary accuracy over training samples (denominator: `n_train_samples`) |
| `cls_acc_train` | `List[float]` | [0, 1] | Celebrity top-1 accuracy over training samples (denominator: `n_train_samples`) |
| `loss_val` | `List[float]` | ≥ 0 | Same as `loss_train` but on validation set |
| `bin_acc_val` | `List[float]` | [0, 1] | Same as `bin_acc_train` but on validation set |
| `cls_acc_val` | `List[float]` | [0, 1] | Same as `cls_acc_train` but on validation set |

All six lists are populated in order; index `e` holds the value for epoch `e` (0-indexed).

---

## ARCHITECTURE

### Initialization (before epoch loop)

```python
loss_train, loss_val = [], []
bin_acc_train, bin_acc_val = [], []
cls_acc_train, cls_acc_val = [], []
history1 = hl.History()   # hiddenlayer History object
canvas1  = hl.Canvas()    # hiddenlayer Canvas object
best_val_loss = float('inf')
no_improve_count = 0
best_model_state = None
prev_bin_loss = 0.0
prev_cls_loss = 0.0
```

`history1` and `canvas1` are created inside the function — they are not passed in. `best_model_state` holds a `copy.deepcopy(model.state_dict())` snapshot taken at the epoch with the lowest validation loss so far; it remains `None` if `early_stopping_patience is None`. `prev_bin_loss` and `prev_cls_loss` carry the per-batch average unweighted losses from the previous epoch's training phase into the next epoch's weight computation; both start as `0.0` (the callable must handle epoch 0 via its own `epoch == 0` guard).

---

### Epoch loop: `for epoch in range(epochs)`

#### A. Training phase

1. `model.train()`
2. Reset per-epoch accumulators to 0:
   - `total_bin_acc_train`, `total_cls_acc_train` (float)
   - `total_count_train`, `n_train_batches`, `total_loss_train`, `total_bin_loss_train`, `total_cls_loss_train` (int / tensor)
3. Compute epoch weight: `cls_w = cls_weight_fn(epoch, prev_bin_loss, prev_cls_loss) if cls_weight_fn is not None else 1.0`
4. For each `(img, label)` in `dataloader_train`:
   1. `img, label = img.to(device), label.to(device)`
   2. `optimizer.zero_grad()`
   3. `bin_logits, cls_logits = model(img)`
      - `bin_logits`: shape `(B, 8)`, raw logits
      - `cls_logits`: shape `(B, 501)`, raw logits
   4. Compute individual and combined losses:
      ```python
      bin_loss = binary_criterion(bin_logits, label[:, :8])
      cls_loss = class_criterion(cls_logits, label[:, 8].long())
      loss = bin_loss + cls_w * cls_loss
      ```
   5. Accumulate losses:
      ```python
      total_loss_train     += loss
      total_bin_loss_train += bin_loss
      total_cls_loss_train += cls_loss
      ```
   6. `loss.backward()` then `optimizer.step()`
   7. Binary accuracy accumulation:
      ```python
      total_bin_acc_train += (
          ((torch.sigmoid(bin_logits) >= 0.5) == label[:, :8].bool())
          .float().mean(dim=1).sum().item()
      )
      ```
      - sigmoid converts logits to probabilities
      - threshold at 0.5 → predicted boolean
      - compare element-wise to ground truth booleans
      - average over 8 attributes per sample → per-sample accuracy in [0, 1]
      - sum over batch samples → scalar
   8. Classification accuracy accumulation:
      ```python
      total_cls_acc_train += (
          (cls_logits.argmax(1) == label[:, 8].long()).sum().item()
      )
      ```
   9. `total_count_train += label.size(0)`; `n_train_batches += 1`
5. After inner loop — compute and append epoch metrics, then update previous-epoch loss state:
   ```python
   avg_loss_train = total_loss_train / n_train_batches
   loss_train.append(avg_loss_train.item())
   bin_acc_train.append(total_bin_acc_train / total_count_train)
   cls_acc_train.append(total_cls_acc_train / total_count_train)
   prev_bin_loss = (total_bin_loss_train / n_train_batches).item()
   prev_cls_loss = (total_cls_loss_train / n_train_batches).item()
   ```

#### B. Validation phase

Runs inside `with torch.no_grad():` — no gradients are computed at any point.

1. `model.eval()`
2. Reset per-epoch accumulators to 0:
   - `total_bin_acc_val`, `total_cls_acc_val` (float)
   - `total_count_val`, `n_val_batches`, `total_loss_val`
3. For each `(img, label)` in `dataloader_val`:
   - Same device move, forward pass, loss computation, and accuracy formulas as training
   - **No** `optimizer.zero_grad()`, `loss.backward()`, or `optimizer.step()`
4. After inner loop — compute and append epoch metrics using `_val` accumulators (same averaging logic as training)

#### C. Logging block (after both phases, still inside epoch loop)

**If `not hparam_tuning`:**
```python
history1.log(
    epoch,
    train_loss=avg_loss_train,
    train_bin_acc=bin_acc_train[-1],
    train_cls_acc=cls_acc_train[-1],
    val_loss=avg_loss_val,
    val_bin_acc=bin_acc_val[-1],
    val_cls_acc=cls_acc_val[-1],
)
with canvas1:
    canvas1.draw_plot([history1["train_loss"], history1["val_loss"]])
    canvas1.draw_plot([history1["train_bin_acc"], history1["val_bin_acc"]])
    canvas1.draw_plot([history1["train_cls_acc"], history1["val_cls_acc"]])
```

**If `hparam_tuning`:**
```python
print(
    f"epoch: {epoch + 1} -> "
    f"BinAcc: {100 * bin_acc_train[-1]:.2f}%, "
    f"ClsAcc: {100 * cls_acc_train[-1]:.2f}%, "
    f"Loss: {avg_loss_train:.6f}",
    end=" --- ",
)
print(
    f"Val_BinAcc: {100 * bin_acc_val[-1]:.2f}%, "
    f"Val_ClsAcc: {100 * cls_acc_val[-1]:.2f}%, "
    f"Val_Loss: {avg_loss_val:.6f}"
)
```
Two separate `print()` calls on the same line: first ends with `end=" --- "`, second uses default newline.

#### D. Early stopping check (after logging, still inside epoch loop)

Runs only when `early_stopping_patience is not None`:

```python
if early_stopping_patience is not None:
    if avg_loss_val.item() < best_val_loss:
        best_val_loss = avg_loss_val.item()
        no_improve_count = 0
        best_model_state = copy.deepcopy(model.state_dict())
    else:
        no_improve_count += 1
        if no_improve_count >= early_stopping_patience:
            break
```

---

### Post-loop weight restoration (after epoch loop, before return)

```python
if best_model_state is not None:
    model.load_state_dict(best_model_state)
```

---

### Return

```python
return loss_train, bin_acc_train, cls_acc_train, loss_val, bin_acc_val, cls_acc_val
```

---

## INVARIANTS

### Output shape
- All six returned lists have the same length, equal to the number of epochs actually run (≤ `epochs`).
- `loss_train[e]`, `loss_val[e]` are Python `float` (extracted via `.item()`).
- `bin_acc_train[e]`, `bin_acc_val[e]`, `cls_acc_train[e]`, `cls_acc_val[e]` are Python `float` in [0, 1].

### Loss averaging
- Loss is averaged by **number of batches** (`n_train_batches` / `n_val_batches`), not by number of samples.
- This makes the loss value independent of batch size changes.

### Accuracy averaging
- Both binary and classification accuracies are averaged by **number of samples** (`total_count_train` / `total_count_val`), not by number of batches.

### Binary accuracy semantics
- Per-sample binary accuracy = fraction of the 8 attributes predicted correctly for that sample (range [0, 1]).
- The epoch metric is the mean of these per-sample scores across all samples — NOT a strict "all 8 correct" metric.

### Classification accuracy semantics
- Strict top-1 accuracy: a sample is correct if and only if `argmax(cls_logits)` equals the ground-truth celebrity ID.

### Tensor types
- `label[:, :8]` is treated as `float32` (no cast needed) for `BCEWithLogitsLoss`.
- `label[:, 8].long()` — the `.long()` cast is mandatory; `CrossEntropyLoss` requires integer targets.

### Model mode
- `model.train()` is set before the training inner loop each epoch.
- `model.eval()` is set inside `torch.no_grad()` before the validation inner loop each epoch.

### No internal construction
- The function does not construct the optimizer, loss criteria, or dataloaders — all are passed in as parameters or (for `device`) available as a global.

### No scheduler, no gradient clipping
- Learning rate is constant (managed entirely by the external optimizer).
- No gradient norm clipping is applied.
- Early stopping is opt-in via `early_stopping_patience` (see below).

### Early stopping semantics
- Improvement is defined as a strictly lower `avg_loss_val.item()` than `best_val_loss` (`<`, not `<=`).
- `no_improve_count` resets to 0 on any strict improvement; otherwise it increments by 1.
- Stopping fires when `no_improve_count >= early_stopping_patience` (i.e., after `patience` consecutive non-improving epochs).
- If `early_stopping_patience is None`, the block is skipped and `best_model_state` is never written.
- If early stopping never fires (all `epochs` complete), the best-seen checkpoint is still restored, because `best_model_state` was written the last time val loss improved.

### hiddenlayer objects
- `history1` and `canvas1` are created fresh each call to `train()` — they do not persist across calls.

### Classification loss weighting
- `cls_weight_fn` is called once per epoch, before the training inner loop, with `(epoch, prev_bin_loss, prev_cls_loss)` where `epoch` is 0-indexed.
- `prev_bin_loss` and `prev_cls_loss` are the per-batch averages of the **unweighted** individual losses from the training phase of the previous epoch (i.e. `binary_criterion(...)` and `class_criterion(...)` averaged over batches, before scaling by `cls_w`).
- At epoch 0, both are `0.0`. The callable must branch on `epoch == 0` to return a default weight (e.g. `1/8`) since the previous-epoch losses carry no information.
- The returned scalar is reused for every batch in both the training and validation phases of that epoch.
- If `cls_weight_fn` is `None`, `prev_bin_loss` and `prev_cls_loss` are still tracked internally but the weight is hardcoded to `1.0`.
- The function may return any non-negative float. Negative weights are not guarded against.
