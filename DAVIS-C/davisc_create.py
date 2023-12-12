import os, glob, cv2, subprocess, numpy as np
import tensorflow_hub as hub
import tensorflow as tf
from PIL import Image
import skimage.color as cl
from skimage.filters import gaussian
from imgaug.augmenters.artistic import stylize_cartoon


DAVIS_VIDEOS = ['bike-packing',
                'blackswan',
                'bmx-trees',
                'breakdance',
                'camel',
                'car-roundabout',
                'car-shadow',
                'cows',
                'dance-twirl',
                'dog',
                'dogs-jump',
                'drift-chicane',
                'drift-straight',
                'goat',
                'gold-fish',
                'horsejump-high',
                'india',
                'judo',
                'kite-surf',
                'lab-coat',
                'libby',
                'loading',
                'mbike-trick',
                'motocross-jump',
                'paragliding-launch',
                'parkour',
                'pigs',
                'scooter-black',
                'shooting',
                'soapbox']


def checkdir(dr):
    if not os.path.exists(dr):
        os.mkdir(dr)


##### ------------------------------------------------------------------------------------------
##### functions from the creation of the ImageNet-C dataset
##### https://github.com/hendrycks/robustness/blob/master/ImageNet-C/create_c/make_imagenet_c.py
##### ------------------------------------------------------------------------------------------
def glass_blur(y, severity=1):
    # sigma, max_delta, iterations
    c = [(0.7, 1, 2), (0.9, 2, 1), (1, 2, 3), (1.1, 3, 2), (1.5, 4, 2)][severity - 1]

    x = np.uint8(gaussian(np.array(y) / 255., sigma=c[0], multichannel=True) * 255)

    # locally shuffle pixels
    for i in range(c[2]):
        for h in range(y.size[1] - c[1], c[1], -1):
            for w in range(y.size[0] - c[1], c[1], -1):
                dx, dy = np.random.randint(-c[1], c[1], size=(2,))
                h_prime, w_prime = h + dy, w + dx
                # swap
                x[h, w], x[h_prime, w_prime] = x[h_prime, w_prime], x[h, w]

    return np.clip(gaussian(x / 255., sigma=c[0], multichannel=True), 0, 1) * 255


def defocus_blur(y, severity=1):
    c = [(3, 0.1), (4, 0.5), (6, 0.5), (8, 0.5), (10, 0.5)][severity - 1]

    x = np.array(y) / 255.
    kernel = disk(radius=c[0], alias_blur=c[1])

    channels = []
    for d in range(3):
        channels.append(cv2.filter2D(x[:, :, d], -1, kernel))
    channels = np.array(channels).transpose((1, 2, 0))  # 3x224x224 -> 224x224x3

    return np.clip(channels, 0, 1) * 255


def gaussian_noise(y, severity=1):
    c = [.08, .12, 0.18, 0.26, 0.38][severity - 1]

    x = np.array(y) / 255.
    return np.clip(x + np.random.normal(size=x.shape, scale=c), 0, 1) * 255
    

def contrast(y, severity=1):
    c = [0.4, .3, .2, .1, .05][severity - 1]

    x = np.array(y) / 255.
    means = np.mean(x, axis=(0, 1), keepdims=True)
    return np.clip((x - means) * c + means, 0, 1) * 255


def brightness(y, severity=1):
    c = [.1, .2, .3, .4, .5][severity - 1]

    x = np.array(y) / 255.
    x = cl.rgb2hsv(x)
    x[:, :, 2] = np.clip(x[:, :, 2] + c, 0, 1)
    x = cl.hsv2rgb(x)

    return np.clip(x, 0, 1) * 255


def saturate(y, severity=1):
    c = [(0.3, 0), (0.1, 0), (2, 0), (5, 0.1), (20, 0.2)][severity - 1]

    x = np.array(y) / 255.
    x = cl.rgb2hsv(x)
    x[:, :, 1] = np.clip(x[:, :, 1] * c[0] + c[1], 0, 1)
    x = cl.hsv2rgb(x)

    return np.clip(x, 0, 1) * 255


def pixelate(y, severity=1):
    c = [0.6, 0.5, 0.4, 0.3, 0.25][severity - 1]

    x = y.resize((int(y.size[0] * c), int(y.size[1] * c)), Image.BOX)
    x = x.resize((y.size[0], y.size[1]), Image.BOX)

    return x


def disk(radius, alias_blur=0.1, dtype=np.float32):
    if radius <= 8:
        L = np.arange(-8, 8 + 1)
        ksize = (3, 3)
    else:
        L = np.arange(-radius, radius + 1)
        ksize = (5, 5)
    X, Y = np.meshgrid(L, L)
    aliased_disk = np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=dtype)
    aliased_disk /= np.sum(aliased_disk)

    # supersample disk to antialias
    return cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)


##### ------------------------------------------------------------------------------------------
##### code borrowed and modified from 
##### Neural Style Transfer Transition Video Processing 
##### By Brycen Westgarth and Tristan Jogminas
##### https://github.com/westgarthb/style-transfer-video-processor
##### ------------------------------------------------------------------------------------------
class Style:

    def __init__(self, style_files):
        os.environ['TFHUB_CACHE_DIR'] = f'./tensorflow_cache'
        self.hub_module = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
        self.style_files = style_files
        self.get_style_info()

    def get_style_info(self):
        self.style = []
        for fn in self.style_files:
            style = cv2.imread(fn)
            style = cv2.cvtColor(style, cv2.COLOR_BGR2RGB)
            style = style / 255.0
            style = tf.cast(tf.convert_to_tensor(style), tf.float32)
            self.style.append(tf.constant(tf.expand_dims(style, axis=0)))

    def _trim_img(self, img, frame_width, frame_height):
        return img[:frame_height, :frame_width]

    def stylize(self,content, style, frame_width, frame_height, preserve_colors = True):
        content = cv2.cvtColor(content, cv2.COLOR_BGR2RGB) / 255.0
        content = tf.cast(tf.convert_to_tensor(content), tf.float32)
        expanded_content = tf.constant(tf.expand_dims(content, axis=0))
        # Apply style transfer
        stylized_img = self.hub_module(expanded_content, style).pop()
        stylized_img = tf.squeeze(stylized_img)
        stylized_img = np.asarray(self._trim_img(stylized_img, frame_width, frame_height))
        if preserve_colors:
            stylized_img = self._color_correct_to_input(content, stylized_img, frame_width, frame_height)    
        stylized_img = cv2.cvtColor(stylized_img, cv2.COLOR_RGB2BGR) * 255.0

        return stylized_img

    def _color_correct_to_input(self, content, generated, frame_width, frame_height):
        # image manipulations for compatibility with opencv
        content = np.array((content * 255.0), dtype=np.float32)
        content = cv2.cvtColor(content, cv2.COLOR_BGR2YCR_CB)
        generated = np.array((generated * 255.0), dtype=np.float32)
        generated = cv2.cvtColor(generated, cv2.COLOR_BGR2YCR_CB)
        generated = self._trim_img(generated, frame_width, frame_height)
        # extract channels, merge intensity and color spaces
        color_corrected = np.zeros(generated.shape, dtype=np.float32)
        color_corrected[:, :, 0] = generated[:, :, 0]
        color_corrected[:, :, 1] = content[:, :, 1]
        color_corrected[:, :, 2] = content[:, :, 2]
        return cv2.cvtColor(color_corrected, cv2.COLOR_YCrCb2BGR) / 255.0


##### ------------------------------------------------------------------------------------------
##### load DAVIS and create DAVIS-C
##### 
##### ------------------------------------------------------------------------------------------
def process_frames(input_folder, output_folder, severity=1):

    trans_names = ['gaussian_noise', 'contrast', 'brightness', 'saturate', 'glass_blur', 'defocus_blur', 'pixelate',
                   'cartoon', 'motion_blur', 'crf_compression', 'style1', 'style2', 'style3', 'style4']

    st = Style(['styles/1.png', 'styles/2.png', 'styles/3.png', 'styles/4.png'])

    print(input_folder)

    input_folder_files = glob.glob(f'{input_folder}/*')
    if len(input_folder_files):
        # Retrieve an image in the input frame dir to get the width
        img = cv2.imread(input_folder_files[0])
        frame_width = img.shape[1]
        frame_height = img.shape[0]

    checkdir(output_folder)

    for t in trans_names:
        checkdir(output_folder+t+"/")
    for t in trans_names:
        checkdir(output_folder+t+"/"+input_folder.split('/')[-2])

    motionblur_size = [0, 0, 2, 3, 4][severity-1]

    framelist = []
    for count, filename in enumerate(sorted(input_folder_files)):

        content_img = cv2.imread(filename) 
        img_pil = Image.fromarray(content_img)
            
        # from IMAGENET-C
        x = gaussian_noise(img_pil, severity=severity)
        cv2.imwrite((output_folder+"gaussian_noise/"+input_folder.split('/')[-2]+'/'+filename.replace(input_folder, '')), x)

        x = contrast(img_pil, severity=severity)
        cv2.imwrite((output_folder+"contrast/"+input_folder.split('/')[-2]+'/'+filename.replace(input_folder, '')), x)

        x = brightness(img_pil, severity=severity)
        cv2.imwrite((output_folder+"brightness/"+input_folder.split('/')[-2]+'/'+filename.replace(input_folder, '')), x)

        x = saturate(img_pil, severity=severity)
        cv2.imwrite((output_folder+"saturate/"+input_folder.split('/')[-2]+'/'+filename.replace(input_folder, '')), x)

        x = glass_blur(img_pil, severity=severity)
        cv2.imwrite((output_folder+"glass_blur/"+input_folder.split('/')[-2]+'/'+filename.replace(input_folder, '')), x)

        x = defocus_blur(img_pil, severity=severity)
        cv2.imwrite((output_folder+"defocus_blur/"+input_folder.split('/')[-2]+'/'+filename.replace(input_folder, '')), x)

        x = pixelate(img_pil, severity=severity)
        cv2.imwrite((output_folder+"pixelate/"+input_folder.split('/')[-2]+'/'+filename.replace(input_folder, '')), np.array(x))

        # not from IMAGENET-C            
        
        # cartoon
        x = stylize_cartoon(content_img,
                            blur_ksize=[1, 1, 1, 3, 5][severity-1],
                            segmentation_size=[1.0, 1.0, 1.0, 1.2, 1.5][severity-1],
                            saturation=[1.0, 1.0, 1.0, 1.5, 2.0][severity-1],
                            edge_prevalence=.8,
                            suppress_edges=True)
        cv2.imwrite((output_folder+"cartoon/"+input_folder.split('/')[-2]+'/'+filename.replace(input_folder, '')), x)
            
        # stylize
        for j in range(len(st.style)):
            x = st.stylize(content_img, st.style[j], frame_width=frame_width, frame_height=frame_height, preserve_colors=severity <= 4)
            cv2.imwrite((output_folder+"style{}".format(j+1)+"/"+input_folder.split('/')[-2]+'/'+filename.replace(input_folder, '')), x)

        # motion blur
        framelist.append(img_pil)
        if len(framelist) > motionblur_size:
            framelist.pop(0)        
        w = 1
        totalw = w
        x = np.array(framelist[0]).astype(np.float32)
        for j in range(1, len(framelist)): 
            w+=1
            x += w*np.array(framelist[j]).astype(np.float32)
            totalw += w
        x /= totalw
        cv2.imwrite((output_folder+"motion_blur/"+input_folder.split('/')[-2]+'/'+filename.replace(input_folder, '')), np.array(x))
        
    # crf-compression - for the whole video with ffmpeg
    subprocess.run("rm out.mp4 out2.mp4", shell=True)
    cmd_str = "ffmpeg -y -framerate 24 -pattern_type glob -i \'{}/*.jpg\' out.mp4".format(input_folder)
    subprocess.run(cmd_str, shell=True)
    crf_value = [0, 0, 40, 45, 50][severity-1]
    cmd_str = "ffmpeg -y -i out.mp4 -vcodec libx265 -crf {} out2.mp4".format(crf_value)
    subprocess.run(cmd_str, shell=True)
    cmd_str = "ffmpeg -y -i out2.mp4 -r 24 -qscale:v 2 -start_number 0 {}/%05d.jpg".format(output_folder+"crf_compression/"+input_folder.split('/')[-2]+'/')
    subprocess.run(cmd_str, shell=True)


if __name__ == "__main__":

    dir_davis = 'DAVIS/DAVIS17/JPEGImages/480p/'
    dir_out_main = 'DAVISC/'

    checkdir(dir_out_main)
    
    for severity in [3, 4, 5]:
        print("severity {}".format(severity), flush=True)

        if severity == 3:
            print("low")
            dir_out = dir_out_main+'low/'
        elif severity == 4:
            print("med")            
            dir_out = dir_out_main+'med/'                    
        elif severity == 5:
            print("high")            
            dir_out = dir_out_main+'high/'            

        checkdir(dir_out)

        for v in DAVIS_VIDEOS:
            process_frames(dir_davis+v+'/', dir_out, severity = severity)
