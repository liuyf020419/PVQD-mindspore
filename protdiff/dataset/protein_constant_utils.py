
ENCODEAA2NUM = {
    "X": -1,
    "A": 0,
    "C": 1,
    "D": 2,
    "E": 3,
    "F": 4,
    "G": 5,
    "H": 6,
    "I": 7,
    "K": 8,
    "L": 9,
    "M": 10,
    "N": 11,
    "P": 12,
    "Q": 13,
    "R": 14,
    "S": 15,
    "T": 16,
    "V": 17,
    "W": 18,
    "Y": 19,
}
ENCODENUM2AA = {v:k for k, v in ENCODEAA2NUM.items()}

PROTEINLETTER3TO1 = {
    "ALA": "A",
    "CYS": "C",
    "ASP": "D",
    "GLU": "E",
    "PHE": "F",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LYS": "K",
    "LEU": "L",
    "MET": "M",
    "ASN": "N",
    "PRO": "P",
    "GLN": "Q",
    "ARG": "R",
    "SER": "S",
    "THR": "T",
    "VAL": "V",
    "TRP": "W",
    "TYR": "Y",
}
PROTEINLETTER1TO3 = {v:k for k, v in PROTEINLETTER3TO1.items()}

#### {"ASX": "ASP", "XAA": "GLY", "GLX": "GLU", "XLE": "LEU", "SEC": "CYS", "PYL": "LYS", "UNK": "GLY"}
non_standardAA = {"ASX": "D", "XAA": "G", "GLX": "E", "XLE": "L", "SEC": "C", "PYL": "K", "UNK": "G", "PTR": "Y", "MSE": "M"}
PROTEINLETTER3TO1.update(non_standardAA)

