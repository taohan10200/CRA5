import os 
os.environ['http_proxy']='http://10.1.11.100:8086/'
os.environ['HTTP_PROXY']='http://10.1.11.100:8086/'
os.environ['https_proxy']='http://10.1.11.100:8086/'
os.environ['HTTPS_PROXY']='http://10.1.11.100:8086/'
from cra5.models.compressai.zoo import bmshj2018_factorized
print(bmshj2018_factorized)
from cra5.models.compressai.zoo import vaeformer_pretrained
net = bmshj2018_factorized(quality=2, pretrained=True).eval()
net = vaeformer_pretrained(quality=268, pretrained=True).eval()
# print(net)