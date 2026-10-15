import math
import re
import struct
from typing import Dict, Any, List

SUSPICIOUS_APIS = [
    "VirtualAlloc", "VirtualAllocEx", "VirtualProtect", "CreateRemoteThread",
    "WriteProcessMemory", "ReadProcessMemory", "OpenProcess", "URLDownloadToFileA",
    "URLDownloadToFileW", "WinExec", "ShellExecuteA", "ShellExecuteW", "RegSetValueExA",
    "RegSetValueExW", "RegCreateKeyExA", "GetAsyncKeyState", "SetWindowsHookExA",
    "BitBlt", "InternetOpenA", "InternetConnectA", "HttpSendRequestA", "WSAStartup",
    "CreateProcessA", "CreateProcessW", "AdjustTokenPrivileges", "LookupPrivilegeValueA"
]

SUSPICIOUS_KEYWORDS = [
    "cmd.exe", "powershell.exe", "reg.exe", "vssadmin", "shadowcopy", "bcdedit",
    "SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Run", "HTTP://", "HTTPS://",
    "BITCOIN", "TOR", ".LOCKED", "PAYMENT", "DECRYPT", "RANSOM"
]

PACKER_SIGNATURES = [b"UPX!", b"ASPack", b"Themida", b"MPRESS", b"PECompact"]

def calculate_entropy(data: bytes) -> float:
    if not data:
        return 0.0
    byte_counts = [0] * 256
    for b in data:
        byte_counts[b] += 1
    
    length = len(data)
    entropy = 0.0
    for count in byte_counts:
        if count == 0:
            continue
        p = count / length
        entropy -= p * math.log2(p)
    return float(round(entropy, 4))

def extract_strings(data: bytes, min_len: int = 4) -> List[str]:
    # Extract ASCII strings
    pattern = bytes(f"[A-Za-z0-9/\\-_:;.,?!=@$%#]{{{min_len},}}", 'ascii')
    return [s.decode('ascii', errors='ignore') for s in re.findall(pattern, data)]

def parse_pe_header_basic(data: bytes) -> Dict[str, Any]:
    # Pure python basic PE header parsing
    is_pe = False
    num_sections = 0
    subsystem = 0
    sections = []
    
    if len(data) > 0x40 and data[:2] == b'MZ':
        pe_offset = struct.unpack('<I', data[0x3C:0x40])[0]
        if len(data) > pe_offset + 24 and data[pe_offset:pe_offset+4] == b'PE\x00\x00':
            is_pe = True
            num_sections = struct.unpack('<H', data[pe_offset+6:pe_offset+8])[0]
            
            # Section headers typically start after optional header
            optional_hdr_size = struct.unpack('<H', data[pe_offset+20:pe_offset+22])[0]
            section_offset = pe_offset + 24 + optional_hdr_size
            
            for i in range(min(num_sections, 16)):
                s_hdr = data[section_offset + i*40 : section_offset + (i+1)*40]
                if len(s_hdr) < 40:
                    break
                sec_name = s_hdr[:8].rstrip(b'\x00').decode('ascii', errors='ignore')
                vsize = struct.unpack('<I', s_hdr[8:12])[0]
                raw_size = struct.unpack('<I', s_hdr[16:20])[0]
                raw_offset = struct.unpack('<I', s_hdr[20:24])[0]
                
                sec_bytes = data[raw_offset : raw_offset + raw_size] if raw_offset + raw_size <= len(data) else b''
                sec_entropy = calculate_entropy(sec_bytes) if sec_bytes else 0.0
                
                sections.append({
                    "name": sec_name or f"sec_{i}",
                    "virtual_size": vsize,
                    "raw_size": raw_size,
                    "entropy": sec_entropy
                })
                
    return {
        "is_pe": is_pe,
        "num_sections": num_sections,
        "sections": sections
    }

def extract_features(data: bytes) -> Dict[str, Any]:
    file_size = len(data)
    entropy = calculate_entropy(data)
    
    # Byte Histogram (first 16 bins aggregate)
    histogram = [0] * 16
    if file_size > 0:
        for b in data:
            histogram[b // 16] += 1
        histogram = [round(h / file_size, 4) for h in histogram]
        
    # Strings Analysis
    strings = extract_strings(data)
    string_count = len(strings)
    printable_ratio = round(sum(len(s) for s in strings) / max(1, file_size), 4)
    avg_string_len = round(sum(len(s) for s in strings) / max(1, string_count), 2)
    
    # Suspicious API matches
    found_apis = []
    text_data = data.decode('ascii', errors='ignore')
    for api in SUSPICIOUS_APIS:
        if api in text_data:
            found_apis.append(api)
            
    # Suspicious Keyword matches
    found_keywords = []
    text_upper = text_data.upper()
    for kw in SUSPICIOUS_KEYWORDS:
        if kw in text_upper:
            found_keywords.append(kw)
            
    # Packer Check
    packer_detected = "None"
    for sig in PACKER_SIGNATURES:
        if sig in data:
            packer_detected = sig.decode('ascii', errors='ignore')
            break
            
    # PE Structure
    pe_info = parse_pe_header_basic(data)
    
    # High entropy section count
    high_entropy_sections = sum(1 for s in pe_info.get("sections", []) if s.get("entropy", 0) > 7.0)
    
    return {
        "file_size": file_size,
        "entropy": entropy,
        "byte_histogram": histogram,
        "string_count": string_count,
        "printable_ratio": printable_ratio,
        "avg_string_len": avg_string_len,
        "suspicious_api_count": len(found_apis),
        "suspicious_apis": found_apis,
        "suspicious_keyword_count": len(found_keywords),
        "suspicious_keywords": found_keywords,
        "packer_detected": packer_detected,
        "is_pe": 1 if pe_info["is_pe"] else 0,
        "num_sections": pe_info["num_sections"],
        "high_entropy_sections": high_entropy_sections,
        "sections": pe_info["sections"]
    }

def get_ml_feature_vector(feat_dict: Dict[str, Any]) -> List[float]:
    """Flattens feature dict into numeric vector for Scikit-Learn ML Model."""
    vector = [
        float(feat_dict["file_size"]),
        float(feat_dict["entropy"]),
        float(feat_dict["string_count"]),
        float(feat_dict["printable_ratio"]),
        float(feat_dict["avg_string_len"]),
        float(feat_dict["suspicious_api_count"]),
        float(feat_dict["suspicious_keyword_count"]),
        float(feat_dict["is_pe"]),
        float(feat_dict["num_sections"]),
        float(feat_dict["high_entropy_sections"]),
        1.0 if feat_dict["packer_detected"] != "None" else 0.0
    ]
    # Add byte histogram (16 features)
    vector.extend(feat_dict.get("byte_histogram", [0.0] * 16))
    return vector

if __name__ == "__main__":
    sample = b"MZ\x90\x00\x03\x00\x00\x00 VirtualAlloc CreateRemoteThread cmd.exe http://malicious.test"
    res = extract_features(sample)
    print("Extracted Features:", res)
    print("Vector:", get_ml_feature_vector(res))
