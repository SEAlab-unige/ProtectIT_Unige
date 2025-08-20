import os
from scapy.all import rdpcap, IP, TCP, UDP, IPv6, Ether
import numpy as np
import struct
from scapy.utils import PcapReader
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
from hashlib import sha256
import traceback
import socket
import psutil
from scapy.packet import Raw
from scapy.compat import raw
from scapy.all import IP, TCP, UDP, Ether, Raw
import gc


# === LABEL MAPPING SECTION ===

# Default: ISCX VPN/NonVPN dataset (active)
label_mapping = {
    "Chat": ["aim_chat", "aimchat", "facebook_chat", "facebookchat", "hangout_chat", "hangouts_chat", "icq_chat", "icqchat", "skype_chat"],
    "Email": ["email","gmail"],
    "VoIP": ["facebook_audio", "hangouts_audio", "skype_audio", "voipbuster"], #"facebook_video", "hangouts_video", "skype_video"],
    "Streaming": ["netflix", "spotify", "vimeo", "youtube", "youtubeHTML5"],
    "File_Transfer": ["ftps_down", "ftps_up", "sftp", "sftpdown", "sftpup", "sftp_down", "sftp_up", "skype_file", "scp"],
    #"P2P": ["bittorrent"],

    "VPN_Chat": ["vpn_aim_chat", "vpn_facebook_chat", "vpn_hangouts_chat", "vpn_icq_chat", "vpn_skype_chat"],
    "VPN_Email": ["vpn_email","vpn_gmail"],
    "VPN_VoIP": ["vpn_facebook_audio", "vpn_hangouts_audio", "vpn_skype_audio", "vpn_voipbuster", "vpn_facebook_video", "vpn_hangouts_video", "vpn_skype_video"],
    "VPN_Streaming": ["vpn_netflix", "vpn_spotify", "vpn_vimeo", "vpn_youtube"],
    "VPN_File_Transfer": ["vpn_ftps", "vpn_sftp", "vpn_skype_files"],
    "VPN_P2P": ["vpn_bittorrent"]
}

# === Optional: Use this for USTC-TFC2016 dataset ===
# label_mapping = {
#     "Benign_BitTorrent": ["bittorrent"],
#     "Benign_Facetime": ["facetime"],
#     "Benign_FTP": ["ftp"],
#     "Benign_Gmail": ["gmail"],
#     "Benign_MySQL": ["mysql"],
#     "Benign_Outlook": ["outlook"],
#     "Benign_Skype": ["skype"],
#     "Benign_SMB": ["smb"],
#     "Benign_Weibo": ["weibo"],
#     "Benign_WorldOfWarcraft": ["worldofwarcraft"],
#
#     "Malware_Cridex": ["cridex"],
#     "Malware_Geodo": ["geodo"],
#     "Malware_Htbot": ["htbot"],
#     "Malware_Miuref": ["miuref"],
#     "Malware_Neris": ["neris"],
#     "Malware_Nsis-ay": ["nsis-ay"],
#     "Malware_Shifu": ["shifu"],
#     "Malware_Tinba": ["tinba"],
#     "Malware_Virut": ["virut"],
#     "Malware_Zeus": ["zeus"]
# }

# === Optional: Use this for QUIC-NetFlow dataset ===
# label_mapping = {
#     "YouTube": ["youtube"],
#     "FileTransfer": ["filetransfer"],
#     "GoogleHangout_Chat": ["googlehangout_chat"],
#     "GoogleHangout_VoIP": ["googlehangout_voip"],
#     "Google_Play_Music": ["google_play_music"]
# }

# === You can define and switch label_mapping manually to match your dataset ===


# Map string labels to integers
label_to_int = {label: i for i, label in enumerate(label_mapping)}
label_to_int["unknown"] = 255  # Assign a valid ubyte value for unknown labels

# Print the label-to-integer mapping
for label, int_value in label_to_int.items():
    print(f"'{label}': {int_value}")

def get_label_from_filename(filename):
    """Assign a label to a file based on its name."""
    filename = filename.lower()  # Ensure case-insensitive matching
    for label, keywords in label_mapping.items():
        if any(keyword in filename for keyword in keywords):
            return label
    return "unknown"

def pad_udp_header(packet):
    """Pad UDP headers to 20 bytes to match TCP headers for deep learning model, without overwriting the payload."""
    if UDP in packet:
        udp_layer = packet[UDP]
        udp_header_len = len(raw(udp_layer))  # Get the length of the raw UDP header

        # UDP headers are typically 8 bytes, so we pad it up to 20 bytes
        if udp_header_len < 20:
            padding_needed = 20 - udp_header_len
            packet[UDP].payload = Raw(b'\x00' * padding_needed) / udp_layer.payload
    return packet

def anonymize_mac(mac_address):
    """Anonymize a MAC address using SHA-256 and return the first 8 characters of the hash."""
    return hashlib.sha256(mac_address.encode()).hexdigest()[:8]


def zero_mac(mac_address):
    """Return a '00:00:00:00:00:00' string to represent a zeroed MAC address."""
    return "00:00:00:00:00:00"


def zero_ip(ip_address):
    """Return a '0.0.0.0' string to represent a zeroed IP."""
    return "0.0.0.0"


def anonymize_ip(ip_address):
    """Anonymize an IPv4 address using SHA-256 and return the first 8 characters of the hash."""
    if '.' in ip_address:  # IPv4 check
        return hashlib.sha256(ip_address.encode()).hexdigest()[:8]
    else:
        return "unknown"


def create_session_key(packet):
    """Create a session key based on original source and destination IP addresses, ports, and protocol."""

    # Ensure that the packet contains an IP layer
    if IP in packet:
        ip_layer = packet[IP]  # Correctly access the IP layer from the packet
    else:
        return None

    # Determine if the packet uses TCP or UDP
    protocol = TCP if TCP in packet else UDP if UDP in packet else None
    if not protocol:
        return None

    # Use the original IP addresses for the session key
    src_ip = ip_layer.src
    dst_ip = ip_layer.dst

    # Sort IP addresses to make the session key order-independent
    ips = sorted([src_ip, dst_ip])

    # Sort ports to make the session key order-independent
    ports = sorted([packet[protocol].sport, packet[protocol].dport])

    # Return the session key as a tuple of sorted IPs, sorted ports, and the protocol name
    return (ips[0], ips[1], ports[0], ports[1], protocol.name)


def extract_packet_data(packet, mac_strategy='remove', ip_strategy='anonymize', zero_ports=False, pad_udp=False):
    """Extracts packet data, removes Ethernet header, and handles IP, MAC addresses, and ports based on strategy.
    mac_strategy: 'remove', 'anonymize', or 'zero'
    Zero_ports False or True ; pad_udp False or True
    ip_strategy: 'anonymize', 'remove', or 'zero'
    """

    # Start by extracting the entire raw packet as bytes
    packet_data = raw(packet)

    # Handle MAC manipulation first
    if Ether in packet:
        eth_layer = packet[Ether]
        if mac_strategy == 'remove':  # Remove the entire Ethernet header (14 bytes for Ethernet II)
            packet_data = packet_data[14:]  # Strip off Ethernet header
        else:  # Handle anonymize or zero strategies for MAC
            if mac_strategy == 'anonymize':
                src_mac_anon = anonymize_mac(eth_layer.src)
                dst_mac_anon = anonymize_mac(eth_layer.dst)
                # MAC addresses are 6-byte binary values, so we need binary data, not string encoding
                src_mac_bin = bytes.fromhex(src_mac_anon)
                dst_mac_bin = bytes.fromhex(dst_mac_anon)
                # Replace the MAC addresses in the raw byte data (first 6 bytes = src MAC, next 6 = dst MAC)
                packet_data = packet_data[:6] + src_mac_bin + packet_data[12:18] + dst_mac_bin + packet_data[18:]
            elif mac_strategy == 'zero':
                src_mac_zero = zero_mac(eth_layer.src)
                dst_mac_zero = zero_mac(eth_layer.dst)
                # MAC addresses are 6-byte binary values
                src_mac_bin = bytes.fromhex(src_mac_zero.replace(':', ''))
                dst_mac_bin = bytes.fromhex(dst_mac_zero.replace(':', ''))
                packet_data = packet_data[:6] + src_mac_bin + packet_data[12:18] + dst_mac_bin + packet_data[18:]

    # Now apply IP address modifications
    if IP in packet:
        ip_layer = packet[IP]
        ip_header_start = 14 if Ether in packet else 0  # Adjust for Ethernet header if present
        ip_src_start = ip_header_start + 12  # Source IP starts 12 bytes after IP header
        ip_dst_start = ip_src_start + 4      # Destination IP starts after source IP (4 bytes later)

        if ip_strategy == 'anonymize':
            src_ip_anon = anonymize_ip(ip_layer.src)
            dst_ip_anon = anonymize_ip(ip_layer.dst)
            # Ensure anonymized IPs are exactly 4 bytes, use first 8 hex characters from anonymization
            src_ip_bin = bytes.fromhex(src_ip_anon[:8])
            dst_ip_bin = bytes.fromhex(dst_ip_anon[:8])
            # Replace the IP addresses in the raw byte data
            packet_data = packet_data[:ip_src_start] + src_ip_bin + packet_data[ip_src_start + 4:ip_dst_start] + dst_ip_bin + packet_data[ip_dst_start + 4:]
        elif ip_strategy == 'remove':
            # Use a unique marker for removed IP addresses (e.g., `b'\xff\xff\xff\xff'`)
            packet_data = packet_data[:ip_src_start] + b'\xff\xff\xff\xff' + packet_data[ip_src_start + 4:ip_dst_start] + b'\xff\xff\xff\xff' + packet_data[ip_dst_start + 4:]
        elif ip_strategy == 'zero':
            # Replace IP addresses with `0.0.0.0` (4 zeroed bytes)
            packet_data = packet_data[:ip_src_start] + b'\x00\x00\x00\x00' + packet_data[ip_src_start + 4:ip_dst_start] + b'\x00\x00\x00\x00' + packet_data[ip_dst_start + 4:]

        # Zero out the TCP or UDP ports if requested
        if (TCP in packet or UDP in packet) and zero_ports:
            # Calculate transport layer start position
            ip_header_len = packet[IP].ihl * 4  # IP header length in bytes
            transport_header_start = 14 + ip_header_len if Ether in packet else ip_header_len

            # Source port starts at the beginning of the transport header (first 2 bytes)
            port_src_start = transport_header_start
            port_dst_start = port_src_start + 2  # Destination port starts after source port

            # Replace the source and destination ports with zero
            packet_data = packet_data[:port_src_start] + b'\x00\x00' + packet_data[port_dst_start:port_dst_start + 2] + b'\x00\x00' + packet_data[port_dst_start + 2:]


    # Optionally pad the UDP header
    if pad_udp and UDP in packet:
        packet = pad_udp_header(packet)  # Pad the UDP header in the actual packet
        packet_data = raw(packet)  # Re-extract the raw data since we modified the UDP layer

    # Return the modified raw packet data
    return packet_data



def is_irrelevant_packet(packet):
    """Check if the packet is irrelevant (e.g., TCP segments with SYN, ACK, FIN flags and no payload, or DNS packets)."""
    if IP not in packet:
        return True
    if TCP in packet and (packet[TCP].flags.S or packet[TCP].flags.F or packet[TCP].flags.A) and len(packet[TCP].payload) == 0:
        return True
    if UDP in packet and packet[UDP].dport == 53:  # DNS typically uses port 53
        return True
    return False

def extract_sessions(pcap_file, length=784):
    """Extract and process sessions efficiently without memory overflow."""
    sessions = {}
    unique_sessions = {}
    original_lengths = []
    truncated = padded = used_full_length = 0

    with PcapReader(pcap_file) as packets:
        for packet in packets:
            if is_irrelevant_packet(packet):
                continue

            session_key = create_session_key(packet)
            if session_key is None:
                continue

            packet_data = extract_packet_data(packet)

            # Append data to session
            sessions.setdefault(session_key, bytearray()).extend(packet_data)

    # De-duplicate sessions
    for session_key, session_data in sessions.items():
        if not session_data:
            continue
        session_hash = hashlib.sha256(session_data).hexdigest()
        if session_hash not in unique_sessions:
            unique_sessions[session_hash] = (session_key, session_data)
            original_lengths.append(len(session_data))

    final_sessions = {k: v[1] for k, v in unique_sessions.items()}

    # Adjust session length
    for session_key, session_data in final_sessions.items():
        session_length = len(session_data)
        if session_length > length:
            truncated += 1
            final_sessions[session_key] = session_data[:length]
        elif session_length < length:
            padded += 1
            final_sessions[session_key] = session_data.ljust(length, b'\x00')
        else:
            used_full_length += 1

    average_length = sum(original_lengths) / len(original_lengths) if original_lengths else 0

    print(f"Sessions Total: {len(final_sessions)}, Truncated: {truncated}, Padded: {padded}, "
          f"Used Full Length: {used_full_length}, Average Original Length: {average_length:.2f}")

    stats = {
        'Sessions Total': len(final_sessions),
        'Truncated': truncated,
        'Padded': padded,
        'Used Full Length': used_full_length,
        'Average Original Length': average_length
    }

    return final_sessions, stats


def extract_sessions_and_label(pcap_file, length=784):
    label_str = get_label_from_filename(os.path.basename(pcap_file))
    label_int = label_to_int.get(label_str, 255)
    sessions, stats = extract_sessions(pcap_file, length)
    labels = [label_int] * len(sessions)
    return sessions, labels, stats


def convert_sessions_to_matrices(sessions):
    """Generator to process session data into 28x28 matrices one at a time."""
    for session_data in sessions.values():
        yield np.array(list(session_data), dtype=np.uint8).reshape(28, 28)


def save_to_idx3(matrices, filename):
    mode = 'ab' if os.path.exists(filename) else 'wb'
    with open(filename, mode) as file:
        if mode == 'wb':
            file.write(struct.pack('>IIII', 2051, 0, 28, 28))  # Header
        for matrix in matrices:
            file.write(matrix.astype(np.uint8).tobytes())
            del matrix  # Free memory


def save_to_idx1(labels, filename):
    mode = 'ab' if os.path.exists(filename) else 'wb'
    with open(filename, mode) as file:
        if mode == 'wb':
            file.write(struct.pack('>II', 2049, 0))  # Header
        for label in labels:
            file.write(struct.pack('>B', label))
            del label  # Free memory


def update_idx3_header(filename, num_items):
    with open(filename, 'r+b') as file:
        file.seek(4)
        file.write(struct.pack('>I', num_items))


def update_idx1_header(filename, num_items):
    with open(filename, 'r+b') as file:
        file.seek(4)
        file.write(struct.pack('>I', num_items))


def process_single_file(pcap_file, idx3_path, idx1_path):
    try:
        sessions, labels, stats = extract_sessions_and_label(pcap_file, length=784)

        # Process and save each session to .idx3 file immediately
        for matrix in convert_sessions_to_matrices(sessions):
            save_to_idx3([matrix], idx3_path)

        # Save labels in .idx1 file immediately
        save_to_idx1(labels, idx1_path)

        # Explicitly free memory
        del sessions, labels
        gc.collect()

    except Exception as e:
        print(f"⚠️ Error processing {pcap_file}: {e}")

        # Ensure stats is always assigned
        stats = {
            'Sessions Total': 0,
            'Truncated': 0,
            'Padded': 0,
            'Used Full Length': 0,
            'Average Original Length': 0
        }

    return stats  # ✅ Now stats is always returned


def main():
    import time

    # === USER NOTE: Set paths to folders containing your PCAP files ===
    # Example: Each directory should contain one or more .pcap files.
    directories = [
        r'/put/your/first/pcap/folder/here',
        r'/put/your/second/pcap/folder/here',
        # Add more paths as needed
    ]

    print("🔍 Scanning all files in directories (assuming all are PCAPs)...")

    # Collect ALL files recursively
    pcap_files = []
    for dir_path in directories:
        for root, _, files in os.walk(dir_path):
            for filename in files:
                full_path = os.path.join(root, filename)
                pcap_files.append(full_path)

    print(f"📂 Found {len(pcap_files)} files (assuming all are PCAPs).")
    for i, path in enumerate(pcap_files):
        print(f"  [{i + 1}] {path}")


    if not pcap_files:
        print("⚠️ No .pcap files found! Please check directory paths.")
        return

        # === USER NOTE: Set your desired output file paths for the IDX3 and IDX1 files ===
        idx3_path = r'/put/your/output/folder/session_output.idx3'  # <-- Change this path
        idx1_path = r'/put/your/output/folder/label_output.idx1'  # <-- Change this path

    # Remove old output files
    if os.path.exists(idx3_path):
        os.remove(idx3_path)
    if os.path.exists(idx1_path):
        os.remove(idx1_path)

    total_sessions = total_truncated = total_padded = total_full_length = total_original_length = 0

    start_time = time.time()

    with ProcessPoolExecutor(max_workers=16) as executor:
        futures = {
            executor.submit(process_single_file, pcap_file, idx3_path, idx1_path): pcap_file
            for pcap_file in pcap_files
        }

        for future in as_completed(futures):
            pcap_file = futures[future]
            try:
                stats = future.result(timeout=600)
                print(f"✅ Finished {pcap_file} — {stats['Sessions Total']} sessions")
            except Exception as e:
                print(f"❌ Failed processing {pcap_file}: {e}")
                continue

            # Accumulate stats
            total_sessions += stats['Sessions Total']
            total_truncated += stats['Truncated']
            total_padded += stats['Padded']
            total_full_length += stats['Used Full Length']
            total_original_length += stats['Average Original Length'] * stats['Sessions Total']

            del stats
            gc.collect()

    end_time = time.time()
    duration = end_time - start_time

    final_average_length = total_original_length / total_sessions if total_sessions else 0

    print(f"\n📊 Final Stats:")
    print(f"  Sessions Total: {total_sessions}")
    print(f"  Truncated: {total_truncated}")
    print(f"  Padded: {total_padded}")
    print(f"  Used Full Length: {total_full_length}")
    print(f"  Average Original Length: {final_average_length:.2f}")
    print(f"  Processing Time: {duration:.2f} seconds")

    update_idx3_header(idx3_path, total_sessions)
    update_idx1_header(idx1_path, total_sessions)



if __name__ == "__main__":
    main()
