import hashlib
import os
from typing import Dict, Optional

def calculate_sha256(file_path: str) -> str:
    """
    Calculate the SHA-256 cryptographic hash of a given file.
    
    Reads the file in binary mode in 64KB chunks to safely handle 
    files of any size without loading the entire binary into memory.
    """
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(65536), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest().lower()


def get_file_hashes(file_path: str) -> Dict[str, str]:
    """
    Calculate MD5, SHA-1, and SHA-256 hashes for a file.
    
    Returns a dictionary containing all three cryptographic hashes.
    """
    md5_hash = hashlib.md5()
    sha1_hash = hashlib.sha1()
    sha256_hash = hashlib.sha256()

    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(65536), b""):
            md5_hash.update(byte_block)
            sha1_hash.update(byte_block)
            sha256_hash.update(byte_block)

    return {
        "md5": md5_hash.hexdigest().lower(),
        "sha1": sha1_hash.hexdigest().lower(),
        "sha256": sha256_hash.hexdigest().lower()
    }
