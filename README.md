# 🔐 ProtectIT_Unige: Efficient Traffic Classification on Resource-Constrained Devices

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

### [`processed_datasets/`](./processed_datasets/)
Contains session-level data generated from various public datasets using different preprocessing strategies.

---

## 🧠 Pipeline Overview

1. **Extract sessions** from `.pcap` traffic using the preprocessing module.
2. **Search and train architectures** using the NAS engine under resource constraints.
3. **Evaluate and select models** based on accuracy and hardware footprint.

---

## 🎯 Objective

The goal is to discover deep learning models that:
- Are accurate for encrypted traffic classification
- Fit the constraints of **edge or embedded devices**, including **microcontrollers**
- Require minimal compute, memory, and storage resources

---

## ⚠️ Note

This repository is part of an academic research project. The corresponding paper is currently under peer review.

---

## 📚 Requirements

- Python 3.x

Install required packages:
```bash
pip install scapy numpy psutil tensorflow keras-flops scikit-learn
```
