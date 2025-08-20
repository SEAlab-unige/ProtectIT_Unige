# 📦 Session Preprocessing Script

This script processes network traffic from `.pcap` files by extracting sessions, anonymizing sensitive fields, and converting them into fixed-size 28×28 matrices. Outputs are saved in MNIST-style `.idx` format, ready for machine learning workflows.

It supports:
- Fixed-length session extraction with padding/truncation
- Anonymization of MACs, IPs, ports, and UDP headers
- Labeling based on filename keywords (configurable mapping)
- Incremental output to `.idx3` (features) and `.idx1` (labels)
- Parallel processing across large `.pcap` datasets

---

### 📂 Dataset Support & Label Mapping

Labels are inferred from filename keywords using a configurable `label_mapping` dictionary defined in the script.

- ✅ Default mapping supports **ISCX VPN/NonVPN** dataset.
- 🔄 You can switch to others (e.g. **USTC-TFC2016**, **QUIC-NetFlow**) by editing the dictionary.
- ⚠️ Files without matching keywords are labeled as `"unknown"` (`255`).

> 📌 To use your own dataset: ensure filenames contain identifiable keywords that match your custom label mapping.


## 🔧 Key Functions

### `anonymize_ip(ip_address)`
Anonymizes IP addresses using SHA-256.
- **Input:** `ip_address` (`str`)
- **Output:** 8-character hash (`str`)

### `create_session_key(packet)`
Generates a session key based on IPs, ports, and transport protocol.
- **Input:** `packet` (`scapy.Packet`)
- **Output:** `(src IP, dst IP, src port, dst port, protocol)`

### `extract_packet_data(packet, ...)`
Extracts the raw packet data and applies preprocessing as configured.

Supported operations include MAC/IP anonymization, port zeroing, and optional UDP header padding.

- **Output:** Raw byte stream (`bytes`)


### `is_irrelevant_packet(packet)`
Filters out control packets and DNS traffic.
- **Output:** `True` if irrelevant, else `False`

### `extract_sessions(pcap_file, length=784)`
Builds sessions from packets, deduplicates them, and normalizes length.
- **Output:** `{session_key: session_bytes}`, plus stats (`dict`)

### `extract_sessions_and_label(pcap_file, length=784)`
Extends `extract_sessions()` by assigning labels based on filename keywords.
- **Output:** `sessions`, `labels`, and `stats`

### `convert_sessions_to_matrices(sessions)`
Converts each session byte stream into a 28×28 matrix (`uint8`).
- **Output:** Generator of `np.ndarray` matrices

### `save_to_idx3(matrices, filename)`
Appends image matrices to an `.idx3` file (incrementally).

### `save_to_idx1(labels, filename)`
Appends session labels to an `.idx1` file (incrementally).

---

## 🚀 Script Workflow

The script scans one or more directories for `.pcap` files, processes each in parallel, and incrementally updates `.idx3` and `.idx1` files with session data and labels.

---

## 📁 Usage

1. Update the `directories` list in the `main()` function to point to your `.pcap` folders.
2. Set the desired output paths for `.idx3` and `.idx1` files.
3. Run the script:
```bash
python your_script.py
```
## ⚙️ Preprocessing Configuration

You can control how sensitive fields are handled during preprocessing:

| Field           | Options                        |
|-----------------|--------------------------------|
| MAC addresses   | `remove`, `zero`, `anonymize`  |
| IP addresses    | `remove`, `zero`, `anonymize`  |
| Ports           | Zeroing: `True` or `False`     |
| UDP Header      | Pad to 20 bytes: `True` or `False` |

These options are set in the `extract_packet_data()` function (used internally). Adjust them to meet privacy or model compatibility requirements.

---

## 📤 Output Files

- `session_output.idx3`: All session matrices (shape: 28×28, dtype: `uint8`)
- `label_output.idx1`: Corresponding labels (dtype: `uint8`)

> ✅ These files are updated **incrementally**, allowing you to process multiple PCAP files in sequence without overwriting previous results.

---

## 📊 Runtime Logging

For each PCAP file, the script reports:

- Total sessions extracted
- Number of truncated or padded sessions
- Average original session length (in bytes)
- Processing time per file

---

## 🧵 Parallel Processing

The script uses Python’s `ProcessPoolExecutor` for parallelism:
```python
with ProcessPoolExecutor(max_workers=16) as executor:
    ...
```

## 📚 Dependencies

- Python 3.x

### External packages (install via pip):

- `scapy`
- `numpy`
- `psutil`

```bash
pip install scapy numpy psutil
```

### Built-in modules (no need to install):

- `os`  
- `struct`  
- `hashlib`  
- `traceback`  
- `socket`  
- `gc`  
- `concurrent.futures`



