import re
import os
import glob
import shlex
import subprocess

path = 'pickle_jar/cavity-100.0_256_100120-zero_0_0-c_Lxy2e1-microz'
all_path = glob.glob(path + '/*')

result_no = []

pattern = r'\d{1,2}'

for i,path in enumerate(all_path):
    _, filename = os.path.split(path)

    if filename == 'metadata.pkl':
        metadata_idx = i
        continue

    no = re.findall(pattern, filename)
    assert len(no) == 1
    no = int(no[0])
    result_no.append(no)

del all_path[metadata_idx]

print(len(all_path))
print(len(result_no))
for i, no in enumerate(result_no):
    print(no, all_path[i])
    

