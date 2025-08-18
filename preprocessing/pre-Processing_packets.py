import os
import struct
import hashlib
from scapy.all import PcapReader, Ether, IP, TCP, UDP, Raw
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback
import socket
import psutil
import gc
from gc import collect
from scapy.all import *
import time
from memory_profiler import profile
import h5py

def print_memory_usage():
    process = psutil.Process(os.getpid())
    print(f"Current memory usage: {process.memory_info().rss / 1024 ** 2:.2f} MB")  # RSS: Resident Set Size

label_mapping = {
    "Chat": ["aim_chat", "aimchat", "facebook_chat", "facebookchat", "hangout_chat", "hangouts_chat", "icq_chat", "icqchat", "skype_chat"],
    "Email": ["email","gmail"],
    "VoIP": ["facebook_audio", "hangouts_audio", "skype_audio", "voipbuster","facebook_video", "hangouts_video", "skype_video"],
    "Streaming": ["netflix", "spotify", "vimeo", "youtube", "youtubeHTML5"],
    "File_Transfer": ["ftps_down", "ftps_up", "sftp", "sftpdown", "sftpup", "sftp_down", "sftp_up", "skype_file", "scp"],
    "P2P": ["torrent01"],

    "VPN_Chat": ["vpn_aim_chat", "vpn_facebook_chat", "vpn_hangouts_chat", "vpn_icq_chat", "vpn_skype_chat"],
    "VPN_Email": ["vpn_email","vpn_gmail"],
    "VPN_VoIP": ["vpn_facebook_audio", "vpn_hangouts_audio", "vpn_skype_audio", "vpn_voipbuster","vpn_facebook_video", "vpn_hangouts_video", "vpn_skype_video"],
    "VPN_Streaming": ["vpn_netflix", "vpn_spotify", "vpn_vimeo", "vpn_youtube"],
    "VPN_File_Transfer": ["vpn_ftps", "vpn_sftp", "vpn_skype_files"],
    "VPN_P2P": ["vpn_bittorrent"]
}

#Map string labels to integers
label_to_int = {label: i for i, label in enumerate(label_mapping)}
label_to_int["unknown"] = 255  # Assign a valid ubyte value for unknown labels
# Print the label to integer mapping
for label, int_value in label_to_int.items():
    print(f"'{label}': {int_value}")

def get_label_from_filename(filename):
    filename = filename.lower()  # Convert filename to lowercase to ensure case-insensitive matching
    # First, check for VPN labels due to their specificity
    for label, keywords in label_mapping.items():
        if label.startswith("VPN_") and any(keyword in filename for keyword in keywords):
            return label
    # Then, check for non-VPN labels
    for label, keywords in label_mapping.items():
        if not label.startswith("VPN_") and any(keyword in filename for keyword in keywords):
            return label
    return "unknown"

def is_valid_ipv4(ip_address):
    try:
        socket.inet_pton(socket.AF_INET, ip_address)
        return True
    except socket.error:
        return False

def is_valid_ipv6(ip_address):
    try:
        socket.inet_pton(socket.AF_INET6, ip_address)
        return True
    except socket.error:
        return False

def anonymize_ip(ip_address, salt="my_secret_salt"):
    if is_valid_ipv4(ip_address):
        hash_val = hashlib.sha256((salt + ip_address).encode()).hexdigest()[:8]
        new_ip_parts = [int(hash_val[i:i+2], 16) % 256 for i in range(0, 8, 2)]
        return f"{new_ip_parts[0]}.{new_ip_parts[1]}.{new_ip_parts[2]}.{new_ip_parts[3]}"
    elif is_valid_ipv6(ip_address):
        hash_val = hashlib.sha256((salt + ip_address).encode()).hexdigest()[:32]
        new_ip_parts = [int(hash_val[i:i+4], 16) % 65536 for i in range(0, 32, 4)]
        return f"{new_ip_parts[0]:x}:{new_ip_parts[1]:x}:{new_ip_parts[2]:x}:{new_ip_parts[3]:x}:{new_ip_parts[4]:x}:{new_ip_parts[5]:x}:{new_ip_parts[6]:x}:{new_ip_parts[7]:x}"
    else:
        return "unknown"

def extract_packet_data(packet, length=1500, anonymize=True):
    if anonymize:
        # Delete the Ethernet header
        if Ether in packet:
            packet = packet[Ether].payload

        # Anonymize IP addresses
        if IP in packet:
            packet[IP].src = anonymize_ip(packet[IP].src)
            packet[IP].dst = anonymize_ip(packet[IP].dst)
            del packet[IP].chksum  # Remove checksum to force recalculation

        # Rebuild the packet from modified layers to ensure consistency
        packet = packet.__class__(bytes(packet))

    # Ensure the packet length is fixed, padding with zeros if necessary
    return bytes(packet)[:length].ljust(length, b'\x00')


def is_irrelevant_packet(packet, exclude_tls_key=False):
    # Exclude TCP connection segments with SYN, ACK, or FIN flags and no payload
    if TCP in packet and (packet[TCP].flags.S or packet[TCP].flags.F or packet[TCP].flags.A) and len(
            packet[TCP].payload) == 0:
        return True

    # Exclude DNS packets (UDP port 53)
    if UDP in packet and packet[UDP].dport == 53:
        return True

    # Optionally exclude TLS key exchange packets (common on TCP port 443)
    if exclude_tls_key and TCP in packet and packet[TCP].dport == 443:
        # Check for TLS handshake messages (ClientHello, ServerHello, etc.)
        if Raw in packet:
            tls_payload = packet[Raw].load
            if tls_payload and tls_payload[0] == 0x16:  # TLS handshake content type
                handshake_type = tls_payload[5]  # Check the type of handshake message
                if handshake_type in [0x01, 0x02]:  # ClientHello (0x01) or ServerHello (0x02)
                    return True

    return False


def process_packets(pcap_file, length=1500, batch_size=2000, exclude_tls_key=False):
    results = []
    try:
        label_str = get_label_from_filename(os.path.basename(pcap_file))
        label_int = label_to_int.get(label_str, 255)
        print(f"Processing file: {pcap_file}, Label: {label_str}")

        packets_data = []
        labels = []
        packet_count = 0

        with PcapReader(pcap_file) as packets:
            for packet in packets:
                if is_irrelevant_packet(packet, exclude_tls_key=exclude_tls_key):
                    continue
                packet_data = extract_packet_data(packet, length=length)
                packets_data.append(packet_data)
                labels.append(label_int)
                packet_count += 1
                del packet  # Explicitly delete to free memory

                if packet_count % batch_size == 0:
                    results.append((packets_data.copy(), labels.copy()))
                    packets_data.clear()
                    labels.clear()
                    gc.collect()  # Collect garbage after processing each batch

            if packets_data:  # Ensure to process remaining packets
                results.append((packets_data.copy(), labels.copy()))

        print(f"Processed {packet_count} packets from {pcap_file}")
    except Exception as e:
        print(f"Error processing {pcap_file}: {e}")
        traceback.print_exc()

    return results


def save_to_file_hdf5(data_filename, label_filename, base_directory, packets_data, labels):
    if not os.path.exists(base_directory):
        os.makedirs(base_directory)

    data_path = os.path.join(base_directory, data_filename)
    label_path = os.path.join(base_directory, label_filename)

    with h5py.File(data_path, 'a') as f_data, h5py.File(label_path, 'a') as f_labels:
        if 'data' not in f_data:
            data_dataset = f_data.create_dataset('data', (0, 1500), dtype=np.uint8, maxshape=(None, 1500), chunks=True)
        else:
            data_dataset = f_data['data']
        if 'labels' not in f_labels:
            labels_dataset = f_labels.create_dataset('labels', (0,), dtype=np.uint8, maxshape=(None,), chunks=True)
        else:
            labels_dataset = f_labels['labels']

        # Ensure packets_data is the correct shape
        try:
            packets_data = np.array([np.frombuffer(p, dtype=np.uint8) for p in packets_data]).reshape(-1, 1500)
        except ValueError as e:
            print(f"Error converting packets_data: {e}")
            return

        data_dataset.resize((data_dataset.shape[0] + packets_data.shape[0], 1500))
        data_dataset[-packets_data.shape[0]:] = packets_data

        labels_dataset.resize((labels_dataset.shape[0] + len(labels),))
        labels_dataset[-len(labels):] = np.array(labels, dtype=np.uint8)


def main():
    # Modify the list below to include the directories you want to use
    # Example:
    # directories = [
    #     r'C:\path\to\your\first\directory',
    #     r'C:\path\to\your\second\directory',
    #     r'C:\path\to\your\third\directory'
    # ]

    directories = [
        r'/media/adel99/Hard Disk/iscx/NonVPN-PCAPs-01',
        r'/media/adel99/Hard Disk/iscx/non vpn 2',
        r'/media/adel99/Hard Disk/iscx/non vpn3',
        r'/media/adel99/Hard Disk/iscx/VPN-PCAPS-01',
        r'/media/adel99/Hard Disk/iscx/VPN-PCAPs-02',
    ]



    pcap_files = [os.path.join(dir_path, filename)
                  for dir_path in directories
                  for filename in os.listdir(dir_path)
                  if filename.endswith('.pcap')]

    # Modify the base directory and filenames below to where you want to save the output files
    # Example:
    # base_directory = r'C:\path\to\your\output\directory'
    # data_filename = 'your_data_file.h5'
    # label_filename = 'your_label_file.h5'
    
    base_directory = r'/home/adel99/Documents/npy_packets'
    data_filename = 'packets_pcap.h5'
    label_filename = 'labels_pcap.h5'

    exclude_tls_key = False  # Set this based on the requirements

    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(process_packets, pcap_file, 1500, 2000, exclude_tls_key): pcap_file for pcap_file in
                   pcap_files}
        for future in as_completed(futures):
            try:
                results = future.result()
                for packets_data, labels in results:
                    save_to_file_hdf5(data_filename, label_filename, base_directory, packets_data, labels)
                    packets_data.clear()
                    labels.clear()
                    gc.collect()
                print("Delaying 10s, Processed:", futures[future])
                time.sleep(10)  # Reduce delay between processing each PCAP file
            except Exception as exc:
                print(f'Error processing {futures[future]}: {exc}')
                traceback.print_exc()


        print("Data processing completed.")
        sys.exit(0)  # Ensure the program exits


if __name__ == "__main__":
    main()