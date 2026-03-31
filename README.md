

# AdaLook

Official implementation of **AdaLook**.

> **Note:** In our paper, the method is named **AdaLook**, while in the codebase it is implemented under the name **MedgazeCoOp**.

---

## Training and Evaluation

### 16-shot Training

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/medgazecoop/few_shot_gaze_consistency.sh data dataname 16 BiomedCLIP
```

### 16-shot Testing

```bash
python parse_test_res.py --directory=output/dataname/shots_16/MedgazeCoOp_BiomedCLIP/nctx4_cscFalse_ctpend --test-log
```

---

### Base-to-Novel Training

#### Train on Base Classes

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/medgazecoop/base2new_gaze_consistency.sh data dataname BiomedCLIP
```

#### Test on Base Classes

```bash
python parse_test_res.py --directory output/base2new/train_base/dataname/shots_16/MedgazeCoOp_BiomedCLIP/nctx4_cscFalse_ctpend --test-log
```

#### Test on Novel Classes

```bash
python parse_test_res.py --directory output/base2new/test_new/dataname/shots_16/MedgazeCoOp_BiomedCLIP/nctx4_cscFalse_ctpend --test-log
```

---

## Dataset

📋 Due to the double-blind review process, we are unable to upload the dataset or provide further details at this time.
🚀 Our full eye-tracking data collection pipeline, the gaze dataset, and checkpoints for all tasks will be released later.

**Stay tuned!!!**

---




