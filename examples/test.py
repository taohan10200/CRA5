byte_data =  b'\x91\xff\xff\xb1\xff\xff\xd1\xff\xb1\xff\xff\x00\xf1\xff\xff\x12'

string_data = byte_data.decode('utf-16')

print(string_data)


number = int.from_bytes(byte_data, byteorder='big', signed=True)

print(number)

byte_data = b'\xd1\xff\xff\xff\xd1\xff\xff\xf1\xfb\xff\x1f\xf9q\xff\xff\xffq\xff\xff\xf1'
 
number = int.from_bytes(byte_data, byteorder='big', signed=True)

print(number)


byte_data = b'\xb1\xff\xff\x91\xff\xff\xd1\xff\xd1\xff\xff\xff\xd1\xff\xff\xb1'

hex_data = byte_data.hex()

print(hex_data)