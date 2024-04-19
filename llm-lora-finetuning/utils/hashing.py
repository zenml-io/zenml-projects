import sys
import hashlib

BUF_SIZE = 65536 

def compute_md5(file_path:str)->str:
    md5 = hashlib.md5()

    with open(file_path, 'rb') as f:
        while True:
            data = f.read(BUF_SIZE)
            if not data:
                break
            md5.update(data)
    return md5.hexdigest()