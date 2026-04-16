# Ube Macchiatos: MambaVision Replication Project

## Project Overview
This project is a critical replication and analysis of MambaVision, the first hybrid Mamba-Transformer vision backbone. We focus on the MambaVision-T (Tiny) variant, aiming to verify its accuracy and efficiency claims, and to perform hypothesis-driven ablations on its hybrid mixer block. We compare it against transformer baselines (e.g., DeiT-Tiny / ViT-Tiny) and evaluate on datasets such as STL-10, ObjectNet, and MedMNIST.

## Key Limitations
- **Computational Cost:** Limited to Tiny variants due to hardware constraints.
- **Hardware Divergence:** Throughput benchmarks may differ due to hardware differences.
- **Implementation Complexity:** Reimplementation may introduce subtle bugs affecting performance.

## Datasets
**Note:** Large datasets are not included in this repository due to GitHub file size limits.

- **STL-10:** [Kaggle link](https://www.kaggle.com/datasets/jessicali9530/stl10)
- **ObjectNet:** [objectnet.dev](https://objectnet.dev/)
- **MedMNIST:** [medmnist.com](https://medmnist.com/)

### Download Instructions
Run the provided script to download and extract STL-10 automatically:

```bash
python download_stl10.py
```

For ObjectNet and MedMNIST, please download manually from the links above and follow the instructions in their documentation.

---

## Directory Structure
- `MambaVision/` - Model code and experiments
- `STL-10/` - Place STL-10 data here (after running the script)
- `download_stl10.py` - Script to download STL-10 

---

## Contact
Team: Ube Macchiatos
