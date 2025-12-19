#!/usr/bin/env python3
"""
Fix broken gzip headers in new_style MTN files.
Converts new_style (broken gzip) to old_style (valid gzip).
"""

import sys
import zlib
import struct
from pathlib import Path

def fix_mtn_header(input_file, output_file=None):
    """
    Read a new_style MTN file, decompress it, and rewrite with valid gzip header.
    """
    
    # Read the file
    with open(input_file, 'rb') as f:
        data = f.read()
    
    print(f"Input file: {input_file}")
    print(f"File size: {len(data)} bytes")
    print(f"Original header (first 10 bytes): {data[:10].hex()}")
    
    # Skip the 10-byte broken header and decompress
    compressed_payload = data[10:]
    
    try:
        # Try to decompress using raw deflate (no gzip wrapper)
        decompressed_data = zlib.decompress(compressed_payload, -zlib.MAX_WBITS)
        print(f"Decompressed size: {len(decompressed_data)} bytes")
    except Exception as e:
        print(f"Error decompressing: {e}")
        return False
    
    # Create valid gzip file
    # Compress with standard gzip
    compressed = zlib.compress(decompressed_data, level=6)
    
    # Build proper gzip structure
    # Header (10 bytes)
    gzip_header = bytes([
        0x1f, 0x8b,  # Magic number
        0x08,        # Compression method (deflate)
        0x00,        # Flags (no extra fields)
        0x00, 0x00, 0x00, 0x00,  # Modification time (0)
        0x00,        # Extra flags
        0x00         # OS (FAT filesystem)
    ])
    
    # Compress the data using raw deflate
    compressor = zlib.compressobj(level=6, method=zlib.DEFLATED, wbits=-zlib.MAX_WBITS)
    compressed_data = compressor.compress(decompressed_data)
    compressed_data += compressor.flush()
    
    # Footer (8 bytes): CRC32 + original size
    crc32 = zlib.crc32(decompressed_data) & 0xffffffff
    size = len(decompressed_data) & 0xffffffff
    gzip_footer = struct.pack('<I', crc32) + struct.pack('<I', size)
    
    # Combine into valid gzip file
    valid_gzip = gzip_header + compressed_data + gzip_footer
    
    # Determine output filename
    if output_file is None:
        path = Path(input_file)
        # Always add _fixed suffix
        new_stem = path.stem + '_fixed'
        output_file = path.parent / (new_stem + path.suffix)
    
    # Write the fixed file
    with open(output_file, 'wb') as f:
        f.write(valid_gzip)
    
    print(f"Fixed header (first 10 bytes): {valid_gzip[:10].hex()}")
    print(f"Output file: {output_file}")
    print(f"Output size: {len(valid_gzip)} bytes")
    print("✓ File fixed successfully!")
    
    return True

def main():
    if len(sys.argv) < 2:
        print("Usage: python fix_mtn_header.py <input.mtn> [output.mtn]")
        print("   If output filename is not provided, will create filename_old_style.mtn")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    fix_mtn_header(input_file, output_file)

if __name__ == "__main__":
    main()
