# Packet Processing Script README

This script processes and analyzes network packet capture (PCAP) files. The main functionalities include anonymizing IP addresses, extracting packet data, filtering irrelevant packets, and saving the processed data in HDF5 format. Additionally, the script utilizes parallel processing to efficiently handle multiple PCAP files.

## Key Functions

### `is_valid_ipv4(ip_address)`
- **Purpose:** Validates IPv4 addresses.
- **Input:** `ip_address` (string)
- **Output:** Boolean

### `is_valid_ipv6(ip_address)`
- **Purpose:** Validates IPv6 addresses.
- **Input:** `ip_address` (string)
- **Output:** Boolean

### `anonymize_ip(ip_address, salt="my_secret_salt")`
- **Purpose:** Anonymizes IP addresses with a salt.
- **Input:** `ip_address` (string), `salt` (string, optional)
- **Output:** Anonymized IP address (string)

### `extract_packet_data(packet, length=1500, anonymize=True)`
- **Purpose:** Extracts and optionally anonymizes packet data.
- **Input:** `packet` (scapy packet), `length` (int, optional), `anonymize` (bool, optional)
- **Output:** Packet data (bytes)

### `is_irrelevant_packet(packet, exclude_tls_key=False)`
- **Purpose:** Identifies irrelevant packets.
- **Input:** `packet` (scapy packet), `exclude_tls_key` (bool, optional)
- **Output:** Boolean

### `process_packets(pcap_file, length=1500, batch_size=2000, exclude_tls_key=False)`
- **Purpose:** Processes packets from a PCAP file.
- **Input:** `pcap_file` (string), `length` (int, optional), `batch_size` (int, optional), `exclude_tls_key` (bool, optional)
- **Output:** List of processed packet data and labels

### `save_to_file_hdf5(data_filename, label_filename, base_directory, packets_data, labels)`
- **Purpose:** Saves processed data and labels to HDF5 files.
- **Input:** `data_filename` (string), `label_filename` (string), `base_directory` (string), `packets_data` (list), `labels` (list)
- **Output:** None

## Main Script

The `main()` function processes multiple directories of PCAP files, extracts and processes packet data, and saves the results in HDF5 format. It uses parallel processing to handle multiple files concurrently, optimizing performance.

### Usage
1. Modify the `directories` list in `main()` to include paths to your PCAP files.
2. Modify `base_directory`, `data_filename`, and `label_filename` to set your output paths and filenames.
3. Run the script: `python your_script.py`

### Example Directories
```python
directories = [
    '/path/to/first/directory',
    '/path/to/second/directory',
    '/path/to/third/directory'
]

### Outputs
- `packets_pcap.h5`: Processed packet data
- `labels_pcap.h5`: Corresponding labels

### Statistics
The script prints statistics about the processed packets, including total packets processed.

### Parallel Processing
The script utilizes the `ProcessPoolExecutor` for parallel processing, with the number of workers set to 8. Adjust the number of workers as needed:
```python
with ProcessPoolExecutor(max_workers=16) as executor:
    # Processing logic
```
### Dependencies
- Python 3
- scapy
- numpy
- h5py
