# Session Processing Script README

This script processes network packet capture (PCAP) files. The main functionalities include anonymizing IP addresses, extracting session data, converting them into matrices, and saving the processed data in IDX file format. Additionally, the script utilizes parallel processing to efficiently handle multiple PCAP files.

## Key Functions

### `anonymize_ip(ip_address)`
- **Purpose:** Anonymizes IPv4 and IPv6 addresses.
- **Input:** `ip_address` (string)
- **Output:** Anonymized IP address (string)

### `create_session_key(packet)`
- **Purpose:** Creates a unique session key based on IP addresses and ports.
- **Input:** `packet` (scapy packet)
- **Output:** Tuple of anonymized IPs and ports

### `extract_packet_data(packet)`
- **Purpose:** Extracts and anonymizes packet data, removing the Ethernet header.
- **Input:** `packet` (scapy packet)
- **Output:** Anonymized packet data (bytes)

### `is_irrelevant_packet(packet)`
- **Purpose:** Identifies irrelevant packets (e.g., TCP SYN/ACK/FIN without payload, DNS packets).
- **Input:** `packet` (scapy packet)
- **Output:** Boolean indicating if the packet is irrelevant

### `extract_sessions(pcap_file, length=784)`
- **Purpose:** Extracts sessions from a PCAP file, normalizes session lengths, and removes duplicates.
- **Input:** `pcap_file` (string), `length` (int)
- **Output:** Dictionary of sessions, statistics

### `extract_sessions_and_label(pcap_file, length=784)`
- **Purpose:** Extracts sessions and labels from a PCAP file.
- **Input:** `pcap_file` (string), `length` (int)
- **Output:** Sessions, labels, statistics

### `convert_sessions_to_matrices(sessions)`
- **Purpose:** Converts session data to 28x28 matrices.
- **Input:** `sessions` (dictionary)
- **Output:** List of 28x28 matrices

### `save_to_idx3(matrices, filename)`
- **Purpose:** Saves matrices to an IDX3 file.
- **Input:** `matrices` (list), `filename` (string)
- **Output:** None

### `save_to_idx1(labels, filename)`
- **Purpose:** Saves labels to an IDX1 file.
- **Input:** `labels` (list), `filename` (string)
- **Output:** None

## Main Script

The `main()` function processes multiple directories of PCAP files, extracts and processes session data, and saves the results in IDX format. It uses parallel processing to handle multiple files concurrently, optimizing performance.

### Usage
1. Modify the `directories` list in `main()` to include paths to your PCAP files.
2. Run the script: `python your_script.py`

### Example Directories
```python
directories = [
    '/path/to/first/directory',
    '/path/to/second/directory',
    '/path/to/third/directory'
]
```
### Outputs
- `all_sessions.idx3`: Processed session matrices
- `all_labels.idx1`: Corresponding labels

### Statistics
The script prints statistics about the processed sessions, including total sessions, truncated, padded, and average original length.

### Parallel Processing
The script utilizes the `ProcessPoolExecutor` for parallel processing, with the number of workers set to 16. Adjust the number of workers as needed:
```python
with ProcessPoolExecutor(max_workers=16) as executor:
    # Processing logic
```
### Dependencies
- Python 3
- scapy
- numpy

