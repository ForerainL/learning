# Project Requirements Alignment

## Data format
- Files are stored as `{date}/{jkey}.parquet`, where `jkey` is the stock symbol.
- Feature columns: `x_000` ... `x_136`.
- Label columns: any column whose name starts with `adjMidRet`.
- Metadata column: `obe_seq_num` (must be preserved for later use).
- Target label: **predict `adjMidRet60s`** using all provided features.

## Data splits
- **Train**: 2022-01-01 ~ 2023-12-31
- **Validation**: 2024-01-01 ~ 2024-03-31
- **Test**: 2024-04-01 ~ 2024-12-31

## Part 1: Data understanding
1. Plot histograms of all features and labels **after normalization**.
2. Plot ACF of normalized features and labels.
3. Run OLS analysis on normalized features and labels.
4. Combine 2/3: identify feature types with higher correlation to labels.

## Part 2: Training pipeline
- Implement an **efficient training DataLoader** to sample feature/label pairs from a very large dataset and assemble them into mini-batches.
- Models to implement: **GRU, LSTM, TCN, Transformer**.
- Start with a **small model (<20k parameters)**.
- **Do not use normalization layers** in the models.
- Start with **window size = 64**.

## Part 3: Evaluation
Compute metrics **per date per stock** and then average:
- **IC**
- **RankIC**
- **Quantile return**: sort model predictions, then compute the difference between average return in the top 10% quantile and bottom 10% quantile.

## Part 4: Hyperparameter tuning
- Use **MSE** loss.
- Choose an **optimizer** and **learning-rate scheduler**.
- Tune at least:
  - learning rate
  - batch size
  - total training steps
  - weight decay
  - dropout / model size (vary width and depth)
- You may try other hyperparameters.
- Report results in a **wiki page** with:
  - optimizer and scheduler
  - search space
  - search algorithm
  - final test metrics
- Answer and interpret:
  - Which model performs best?
  - Best hyperparameters per model?
  - Relationship between hyperparameters and performance?
