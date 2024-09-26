    
import struct
from pathlib import Path

def filesize(filepath: str) -> int:
    if not Path(filepath).is_file():
        raise ValueError(f'Invalid file "{filepath}".')
    return Path(filepath).stat().st_size

def write_uints(fd, values, fmt=">{:d}I"):
    fd.write(struct.pack(fmt.format(len(values)), *values))
    return len(values) * 4

def write_bytes(fd, values, fmt=">{:d}s"):
    if len(values) == 0:
            return
    fd.write(struct.pack(fmt.format(len(values)), values))
    
    return len(values) * 1

def read_uints(fd, n, fmt=">{:d}I"):
    sz = struct.calcsize("I")
    return struct.unpack(fmt.format(n), fd.read(n * sz))


def read_uchars(fd, n, fmt=">{:d}B"):
    sz = struct.calcsize("B")
    return struct.unpack(fmt.format(n), fd.read(n * sz))

def read_bytes(fd, n, fmt=">{:d}s"):
    sz = struct.calcsize("s")
    return struct.unpack(fmt.format(n), fd.read(n * sz))[0]


def save_binary_code(self, time, output):
    year = time.split('-')[0]
    file_url=f'{year}/{time}.bin'
    
    with Path(file_url).open("wb") as f:

        out_strings = output["strings"]
        shape = output["z_shape"]
        bytes_cnt = 0
        bytes_cnt = write_uints(f, (shape[0], shape[1], len(out_strings)))
        
        for s in out_strings:
            # import pdb
            # pdb.set_trace()
            print(len(s))
            bytes_cnt += write_uints(f, (len(s),))
            bytes_cnt += write_bytes(f, s)
        # return bytes_cnt

    # self.s3_client.upload_file('/tmp/tmp.bin', object_name=file_url) 

    size = filesize('/tmp/tmp.bin')
    bpp = float(size) * 8 / ( out["x_shape"][1]*out["x_shape"][2]*out["x_shape"][3])
    print(bpp)