import os
from scapy.all import rdpcap, IP, TCP, UDP, IPv6, Ether, Raw
import numpy as np
import struct
from scapy.utils import PcapReader
from concurrent.futures import ProcessPoolExecutor
import hashlib
from hashlib import sha256
import traceback
import socket
import psutil

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

def anonymize_ip(ip_address):
    if '.' in ip_address:  # IPv4 check
        return hashlib.sha256(ip_address.encode()).hexdigest()[:8] #anonymyze ipv4 address
    elif ':' in ip_address:  # IPv6 check
        return hashlib.sha256(ip_address.encode()).hexdigest()[:32] # anonymyze ipv6 address
    else:
        return "unknown"

def create_session_key(packet):
    if IP in packet:
        ip_layer = IP
    elif IPv6 in packet:
        ip_layer = IPv6
    else:
        return None

    protocol = TCP if TCP in packet else UDP if UDP in packet else None
    if not protocol:
        return None

    ips = sorted([anonymize_ip(packet[ip_layer].src), anonymize_ip(packet[ip_layer].dst)])
    ports = sorted([packet[protocol].sport, packet[protocol].dport])

    return (ips[0], ips[1], ports[0], ports[1], protocol.name)

def extract_packet_data(packet):
    """Extracts packet data, removing the Ethernet header and anonymizing the IP addresses."""
    # Remove Ethernet header
    if Ether in packet:
        packet = packet[Ether].payload

    # Ensure we're working with an IP or IPv6 packet
    if IP in packet or IPv6 in packet:
        if IP in packet:
            ip_layer = IP
        elif IPv6 in packet:
            ip_layer = IPv6

        packet_data = bytes(packet[ip_layer])

        # Anonymize IP addresses within the packet data
        src_ip_anon = anonymize_ip(packet[ip_layer].src)
        dst_ip_anon = anonymize_ip(packet[ip_layer].dst)

        packet_data = packet_data.replace(packet[ip_layer].src.encode(), src_ip_anon.encode())
        packet_data = packet_data.replace(packet[ip_layer].dst.encode(), dst_ip_anon.encode())

        return packet_data

    # If it's not an IP or IPv6 packet, return the raw bytes
    return bytes(packet)

def is_irrelevant_packet(packet):
    """Check if the packet is irrelevant (e.g., TCP segments with SYN, ACK, FIN flags and no payload, or DNS packets)."""
    if IP not in packet and IPv6 not in packet:
        return True
    if TCP in packet and (packet[TCP].flags.S or packet[TCP].flags.F or packet[TCP].flags.A) and len(packet[TCP].payload) == 0:
        return True
    if UDP in packet and packet[UDP].dport == 53:
        return True
    return False

def extract_sessions(pcap_file, length=784):
    sessions = {}
    unique_sessions = {}
    original_lengths = []

    with PcapReader(pcap_file) as packets:
        for packet in packets:
            if is_irrelevant_packet(packet):
                continue

            if (IP in packet or IPv6 in packet) and (TCP in packet or UDP in packet):
                session_key = create_session_key(packet)
                if session_key is None:
                    continue
                packet_data = extract_packet_data(packet)
                sessions.setdefault(session_key, bytearray()).extend(packet_data)

    # Process for unique sessions
    for session_key, session_data in sessions.items():
        if not session_data: #skip empty sessions
            continue

        session_hash = hashlib.sha256(session_data).hexdigest()
        if session_hash not in unique_sessions:
            unique_sessions[session_hash] = (session_key, session_data)
            original_lengths.append(len(session_data)) # Capture original length for unique sessions

    # Preparing final sessions after removing duplicates
    final_sessions = {k: v[1] for k, v in unique_sessions.items()}

    # Adjust counters and normalize session data
    truncated, padded, used_full_length = 0, 0, 0
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

    # Compute the average original length
    average_length = sum(original_lengths) / len(original_lengths) if original_lengths else 0

    #print statistics
    print(f"Sessions Total: {len(final_sessions)}, Truncated: {truncated}, Padded: {padded}, Used Full Length: {used_full_length}, Average Original Length: {average_length:.2f}")

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
    label_int = label_to_int.get(label_str, 255)  # Use 255 for 'unknown' labels
    sessions, stats = extract_sessions(pcap_file, length)
    labels = [label_int] * len(sessions)  # Associate each session with the integer label
    return sessions, labels, stats

def convert_sessions_to_matrices(sessions):
    """Convert sessions to 28x28 matrices."""
    return [np.array(list(session_data)).reshape(28, 28) for session_data in sessions.values()]


def save_to_idx3(matrices, filename):
    """Save matrices to IDX3 file format."""
    with open(filename, 'wb') as file:
        # IDX3 file header information
        magic_number = 2051  # Magic number for IDX3 format
        num_images = len(matrices)
        rows = 28
        cols = 28

        # Write header
        file.write(struct.pack('>IIII', magic_number, num_images, rows, cols))

        # Write image data
        for matrix in matrices:
            file.write(matrix.astype(np.uint8).tobytes())

# Function to save labels in IDX1 format
def save_to_idx1(labels, filename):
    with open(filename, 'wb') as file:
        magic_number = 2049
        num_items = len(labels)

        # Write header
        file.write(struct.pack('>II', magic_number, num_items))

        # Write label data
        for label in labels:
            file.write(struct.pack('>B', label))

def process_single_file(pcap_file):
    """Process a single pcap file to extract sessions and labels, then convert to matrices."""
    sessions, labels, stats = extract_sessions_and_label(pcap_file, length=784)
    matrices = convert_sessions_to_matrices(sessions)
    return matrices, labels,stats # Return the same number of values


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


    total_sessions, total_truncated, total_padded, total_full_length, total_original_length = 0, 0, 0, 0, 0
    all_sessions, all_labels = [], []

    with ProcessPoolExecutor(max_workers=16) as executor:
        for matrices, labels, stats in executor.map(process_single_file, pcap_files):
            all_sessions.extend(matrices)
            all_labels.extend(labels)
            total_sessions += stats['Sessions Total']
            total_truncated += stats['Truncated']
            total_padded += stats['Padded']
            total_full_length += stats['Used Full Length']
            total_original_length += stats['Average Original Length'] * stats['Sessions Total']  # Weighted sum for average calculation


    # Calculate the final average original length
    final_average_length = total_original_length / total_sessions if total_sessions else 0

    print(f"Final Stats: Sessions Total: {total_sessions}, Truncated: {total_truncated}, Padded: {total_padded}, Used Full Length: {total_full_length}, Average Original Length: {final_average_length:.2f}")

    # Save the processed sessions and labels
    # Modify the paths below to where you want to save the output files
    # Example:
    # save_to_idx3(all_sessions, r'C:\path\to\your\output\all_sessions.idx3')
    # save_to_idx1(all_labels, r'C:\path\to\your\output\all_labels.idx1')

    save_to_idx3(all_sessions, r'/home/adel99/Documents/idx/all_sessions_pcap_no mac_ipv4ipv6.idx3')
    save_to_idx1(all_labels, r'/home/adel99/Documents/idx/all_labels_pcap_no mac_ipv4ipv6.idx1')

    # verification informations
    print(f"Total number of sessions extracted across all files: {len(all_sessions)}")
    print(f"Total number of labels extracted across all files: {len(all_labels)}")

    if all_labels:
        print(f"Unique labels extracted: {set(all_labels)}")
    else:
        print("No labels extracted.")


if __name__ == "__main__":
    main()