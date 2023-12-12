from argparse import ArgumentParser
import gdown
import os
import tarfile
import urllib.request
import zipfile


"""
Arguments loading
"""
parser = ArgumentParser()
parser.add_argument('--root_dir', default='datasets/video_segmentation')
args = parser.parse_args()

##### DAVIS 2017 #########
davis_dir = os.path.join(args.root_dir, 'DAVIS/2017')
os.makedirs(davis_dir, exist_ok=True)

print('Downloading DAVIS 2017 trainval...')
url = 'https://drive.google.com/uc?id=1kiaxrX_4GuW6NmiVuKGSGVoKGWjOdp6d'
zip_filename = os.path.join(davis_dir, 'DAVIS-2017-trainval-480p.zip')
gdown.download(url, output=zip_filename, quiet=False)
with zipfile.ZipFile(zip_filename, 'r') as zip_file:
    zip_file.extractall(davis_dir)
os.rename(os.path.join(davis_dir, 'DAVIS'), os.path.join(davis_dir, 'trainval'))
os.remove(zip_filename)

print('Downloading DAVIS 2017 testdev...')
url = 'https://drive.google.com/uc?id=1fmkxU2v9cQwyb62Tj1xFDdh2p4kDsUzD'
zip_filename = os.path.join(davis_dir, 'DAVIS-2017-test-dev-480p.zip')
gdown.download(url, output=zip_filename, quiet=False)
with zipfile.ZipFile(zip_filename, 'r') as zip_file:
    zip_file.extractall(davis_dir)
os.rename(os.path.join(davis_dir, 'DAVIS'), os.path.join(davis_dir, 'test-dev'))
os.remove(zip_filename)


##### YouTube VOS 2018 #########
print('Downloading YouTubeVOS2018 val...')
youtube_dir = os.path.join(args.root_dir, 'YouTube2018')
os.makedirs(youtube_dir, exist_ok=True)
url = 'https://drive.google.com/uc?id=1-QrceIl5sUNTKz7Iq0UsWC6NLZq7girr'
zip_filename = os.path.join(youtube_dir, 'valid.zip')
gdown.download(url, output=zip_filename, quiet=False)
with zipfile.ZipFile(zip_filename, 'r') as zip_file:
    zip_file.extractall(youtube_dir)
os.remove(zip_filename)

print('Downloading YouTubeVOS2018 all frames valid...')
youtube_all_frames_dir = os.path.join(args.root_dir, 'YouTube2018/all_frames')
os.makedirs(youtube_all_frames_dir, exist_ok=True)
url = 'https://drive.google.com/uc?id=1yVoHM6zgdcL348cFpolFcEl4IC1gorbV'
zip_filename = os.path.join(youtube_all_frames_dir, 'valid.zip')
gdown.download(url, output=zip_filename, quiet=False)
with zipfile.ZipFile(zip_filename, 'r') as zip_file:
    zip_file.extractall(youtube_all_frames_dir)
os.remove(zip_filename)


##### DAVIS-C #########
print('Downloading DAVIS-C ...')
davisc_dir = os.path.join(args.root_dir, 'DAVIS-C')
os.makedirs(davisc_dir, exist_ok=True)
url = "http://ptak.felk.cvut.cz/personal/toliageo/share/davisc/davisc.tar.gz"
tar_filename = os.path.join(davisc_dir, "davisc.tar.gz")  # Name of the output file
urllib.request.urlretrieve(url, tar_filename)
with tarfile.open(tar_filename, 'r:gz') as tar_file:
    tar_file.extractall(davisc_dir)
os.remove(tar_filename)


##### MOSE #########
mose_dir = os.path.join(args.root_dir, 'MOSE')
os.makedirs(mose_dir, exist_ok=True)

print('Downloading MOSE valid ...')
url="https://drive.google.com/uc?id=1yFoacQ0i3J5q6LmnTVVNTTgGocuPB_hR"
tar_filename = os.path.join(mose_dir, 'valid.tar.gz')
gdown.download(url, tar_filename, quiet=False)
with tarfile.open(tar_filename, 'r:gz') as tar_file:
    tar_file.extractall(mose_dir)
os.remove(tar_filename)

print('Downloading MOSE train ...')
url="https://drive.google.com/uc?id=16Ns7a_frLaCo2ug18UIUkzVYFQqyd4N0"
tar_filename = os.path.join(mose_dir, 'train.tar.gz')
gdown.download(url, tar_filename, quiet=False)
with tarfile.open(tar_filename, 'r:gz') as tar_file:
    tar_file.extractall(mose_dir)
os.remove(tar_filename)