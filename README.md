# 🔐 HW-NAS for Encrypted Traffic Classification on Resource-Constrained Devices

This repository provides a complete pipeline for session-based traffic classification, combining custom packet preprocessing with Neural Architecture Search (NAS) under hardware constraints.

It is designed for research and deployment in embedded or edge environments, where model size, memory, and compute efficiency matter.

---

## 📦 Repository Structure

### [`preprocessing/`](./preprocessing/)
Processes raw `.pcap` network traffic into fixed-length session representations.  
Supports flexible handling of IP/MAC fields, ports, and UDP headers with parallelized processing.  
➡️ See [`README.md`](./preprocessing/)

### [`nas_optimization/`](./nas_optimization/)
Performs hardware-constrained NAS to discover deep learning architectures optimized for low-resource devices.  
Supports proxy/full training, mutation-based evolution, and performance-aware selection.  
➡️ See [`README.md`](./nas_optimization/)

### 📁 Processed Datasets (optional)
Preprocessed session-level datasets (`.idx3` / `.idx1`) used in our experiments  
are available in the [**GitHub Releases**](https://github.com/SEAlab-unige/ProtectIT_Unige/releases).

If you prefer to use your own `.pcap` traffic, use the [`preprocessing/`](./preprocessing/) module to generate compatible inputs.
---

## 🧠 Pipeline Overview

1. **Extract sessions** from `.pcap` traffic using the preprocessing module.
2. **Search and train architectures** using the NAS engine under RAM/Flash/FLOPs constraints.
3. **Evaluate and select models** based on accuracy and hardware footprint.

---

## 🎯 Objective

The goal is to discover deep learning models that:
- Are accurate for encrypted traffic classification
- Fit the constraints of **edge or embedded devices**, including **microcontrollers**
- Require minimal compute, memory, and storage resources

---

## 🚀 Quick Start

1. **Clone the repository**
```bash
git clone https://github.com/SEAlab-unige/ProtectIT_Unige.git
cd ProtectIT_Unige
```

2. **Get the data**  
   - Option A: Download preprocessed `.idx3` / `.idx1` files from the  
     [GitHub Releases](https://github.com/SEAlab-unige/ProtectIT_Unige/releases)  
   - Option B: Preprocess your own raw `.pcap` files:

```bash
cd preprocessing
python session_preprocessing.py
```

3. **Run hardware-constrained NAS**

```bash
cd nas_optimization
python B01_NAS.py
```

---

## 📄 Citation

If you use this code, please cite:
```bash
@article{chehade2026hardware,
  title={Hardware-Aware Neural Architecture Search for Encrypted Traffic Classification on Resource-Constrained Devices},
  author={Chehade, Adel and Ragusa, Edoardo and Gastaldo, Paolo and Zunino, Rodolfo},
  journal={IEEE Transactions on Network and Service Management},
  year={2026},
  publisher={IEEE}
}
```
---

## 📚 Requirements

- Python 3.x

Install required packages:
```bash
pip install scapy numpy psutil tensorflow keras-flops scikit-learn
```
