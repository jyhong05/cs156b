# Week 1 (Rudimentary Only)

This setup intentionally does only basic Week 1 work:
- read train/test CSVs
- inspect label distributions and uncertainty/missingness
- sample-check image paths
- compare basic loss formulas using synthetic logits

## Run in HPC terminal

```bash
cd /path/to/your/cs156b/repo
bash run_week1_hpc.sh
```

## Output files

- `week1_report.txt`
- `week1_raw_label_stats.csv`
- `week1_policy_stats.csv`

## Notes

- No model training is performed here.
- No architecture search/HPO/advanced experiments are included.
- This is intended to be a quick foundation before Week 2.
