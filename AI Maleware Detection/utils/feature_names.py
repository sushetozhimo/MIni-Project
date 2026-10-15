"""
Utility to provide descriptive feature names and categories for the
EMBER 2017 v2 / EMBER 2018 / BODMAS 2,381-dimensional feature vectors.
"""

def get_feature_metadata(index: int):
    """
    Returns (feature_name, category) for a feature index in range [0, 2380].
    """
    if 0 <= index <= 255:
        return f"Byte Histogram [0x{index:02X}]", "Byte Histogram"
    
    elif 256 <= index <= 511:
        bin_idx = index - 256
        return f"Byte Entropy Histogram [Bin {bin_idx}]", "Byte Entropy"
    
    elif 512 <= index <= 615:
        str_idx = index - 512
        if str_idx == 0:
            return "String Count (numstrings)", "Strings"
        elif str_idx == 1:
            return "String Average Length", "Strings"
        elif 2 <= str_idx <= 97:
            char_code = (str_idx - 2) + 0x20
            char_repr = chr(char_code) if 0x20 <= char_code <= 0x7E else f"0x{char_code:02X}"
            return f"Printable Char Dist ['{char_repr}']", "Strings"
        elif str_idx == 98:
            return "Printable Characters Count", "Strings"
        elif str_idx == 99:
            return "String Entropy", "Strings"
        elif str_idx == 100:
            return "File Paths Count in Strings", "Strings"
        elif str_idx == 101:
            return "URLs Count in Strings", "Strings"
        elif str_idx == 102:
            return "Registry Keys Count in Strings", "Strings"
        elif str_idx == 103:
            return "MZ Header Occurrences in Strings", "Strings"
        else:
            return f"String Feature [{str_idx}]", "Strings"
            
    elif 616 <= index <= 625:
        gen_idx = index - 616
        gen_names = [
            "File Size", "Virtual Size", "Has Debug Info", "Exports Count",
            "Imports Count", "Has Relocations", "Has Resources",
            "Has Digital Signature", "Has TLS", "Symbols Count"
        ]
        return f"General PE [{gen_names[gen_idx]}]", "General PE"
        
    elif 626 <= index <= 687:
        hdr_idx = index - 626
        if hdr_idx == 0:
            return "COFF Timestamp", "PE Header"
        elif 1 <= hdr_idx <= 10:
            return f"COFF Machine Hash [{hdr_idx-1}]", "PE Header"
        elif 11 <= hdr_idx <= 20:
            return f"COFF Characteristics Hash [{hdr_idx-11}]", "PE Header"
        elif 21 <= hdr_idx <= 30:
            return f"Subsystem Hash [{hdr_idx-21}]", "PE Header"
        elif 31 <= hdr_idx <= 40:
            return f"DLL Characteristics Hash [{hdr_idx-31}]", "PE Header"
        elif 41 <= hdr_idx <= 50:
            return f"Magic Hash [{hdr_idx-41}]", "PE Header"
        else:
            opt_names = [
                "Major Image Version", "Minor Image Version",
                "Major Linker Version", "Minor Linker Version",
                "Major OS Version", "Minor OS Version",
                "Major Subsystem Version", "Minor Subsystem Version",
                "Size of Code", "Size of Headers", "Size of Heap Commit"
            ]
            sub = hdr_idx - 51
            name = opt_names[sub] if sub < len(opt_names) else f"Header Opt [{sub}]"
            return f"PE Header [{name}]", "PE Header"
            
    elif 688 <= index <= 942:
        sec_idx = index - 688
        if 0 <= sec_idx <= 4:
            gen_sec = [
                "Total Sections Count", "Non-Zero Size Sections Count",
                "Zero-Size Sections Count", "Executable Sections Count",
                "Writable Sections Count"
            ]
            return f"Section Stat [{gen_sec[sec_idx]}]", "Section"
        elif 5 <= sec_idx <= 54:
            return f"Section Sizes Hash [{sec_idx-5}]", "Section"
        elif 55 <= sec_idx <= 104:
            return f"Section Entropy Hash [{sec_idx-55}]", "Section"
        elif 105 <= sec_idx <= 154:
            return f"Section Virtual Size Hash [{sec_idx-105}]", "Section"
        elif 155 <= sec_idx <= 204:
            return f"Entry Section Name Hash [{sec_idx-155}]", "Section"
        elif 205 <= sec_idx <= 254:
            return f"Entry Section Characteristics Hash [{sec_idx-205}]", "Section"
        else:
            return f"Section Feature [{sec_idx}]", "Section"
            
    elif 943 <= index <= 2222:
        imp_idx = index - 943
        if imp_idx < 256:
            return f"Imported Library Hash [{imp_idx}]", "Imports"
        else:
            return f"Imported Function (Lib:Func) Hash [{imp_idx-256}]", "Imports"
            
    elif 2223 <= index <= 2350:
        exp_idx = index - 2223
        return f"Exported Function Hash [{exp_idx}]", "Exports"
        
    elif 2351 <= index <= 2380:
        dd_idx = index - 2351
        dd_names = [
            "Export Table RVA", "Export Table Size",
            "Import Table RVA", "Import Table Size",
            "Resource Table RVA", "Resource Table Size",
            "Exception Table RVA", "Exception Table Size",
            "Certificate Table RVA", "Certificate Table Size",
            "Base Relocation Table RVA", "Base Relocation Table Size",
            "Debug Directory RVA", "Debug Directory Size",
            "Architecture Directory RVA", "Architecture Directory Size",
            "Global Ptr Directory RVA", "Global Ptr Directory Size",
            "TLS Directory RVA", "TLS Directory Size",
            "Load Config Directory RVA", "Load Config Directory Size",
            "Bound Import Directory RVA", "Bound Import Directory Size",
            "IAT Directory RVA", "IAT Directory Size",
            "Delay Import Directory RVA", "Delay Import Directory Size",
            "CLR Runtime Header RVA", "CLR Runtime Header Size"
        ]
        name = dd_names[dd_idx] if dd_idx < len(dd_names) else f"Data Directory [{dd_idx}]"
        return f"Data Directory [{name}]", "Data Directories"
        
    else:
        return f"Unknown Feature [{index}]", "Unknown"

def get_feature_name(index: int) -> str:
    name, _ = get_feature_metadata(index)
    return name

def get_feature_category(index: int) -> str:
    _, cat = get_feature_metadata(index)
    return cat
