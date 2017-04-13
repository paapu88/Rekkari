
import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from scipy import ndimage
import cv2
import random

class GetCharatersForTraining():
    """
    produce positive training set for haar cascade to recognize a character in a finnisn licence plate
    """

    def __init__(self):
        self.font_height = 32
        self.output_height = 48
        self.chars = 'ABCDEFGHIJKLMNOPRSTUVXYZ0123456789'
        self.char_shift_y = -5
        #for noise
        self.sigma=0.1
        self.angle=0
        self.salt_amount=0.1



    def getMinAndMaxY(self, a, thr=0.5):
        """find the value in Y where image starts"""
        minY = None
        maxY = None
        for iy in range(a.shape[0]):
            for ix in range(a.shape[1]):
                if a[iy,ix]> thr:
                    minY = iy
                    break
        for iy in reversed(range(a.shape[0])):
            for ix in range(a.shape[1]):
                if a[iy,ix]> thr:
                    maxY = iy
                    break
        return minY, maxY

    def getMinAndMaxX(self, a, thr=0.5):
        """find the value in Y where image starts"""
        minX = None
        maxX = None
        for ix in range(a.shape[1]):
            for iy in range(a.shape[0]):
                if a[iy,ix]> thr:
                    minX = ix
                    break
        for ix in reversed(range(a.shape[1])):
            for iy in range(a.shape[0]):
                if a[iy,ix]> thr:
                    maxX = ix
                    break
        return minX, maxX





    def noisy(self, noise_typ, image):
        """
        Parameters
        ----------
        image : ndarray
            Input image data. Will be converted to float.
        mode : str
            One of the following strings, selecting the type of noise to add:

            'gauss'     Gaussian-distributed additive noise.
            'poisson'   Poisson-distributed noise generated from the data.
            'sp'       Replaces random pixels with 0 or 1.
            'speckle'   Multiplicative noise using out = image + n*image,where
                        n,is uniform noise with specified mean & variance.
        """

        if noise_typ == "gauss":
            row, col = image.shape
            mean = max(0,np.mean(image))
            # var = 0.1
            # sigma = var**0.5
            print ("M",mean, self.sigma)
            gauss = np.random.normal(mean, self.sigma, (row, col))
            gauss = gauss.reshape(row, col)
            noisy = image + gauss
            return noisy
        elif noise_typ == "sp":
            row, col = image.shape
            s_vs_p = 0.5
            out = image
            # Salt mode
            num_salt = np.ceil(self.salt_amount * image.size * s_vs_p)
            coords = [np.random.randint(0, i - 1, int(num_salt))
                      for i in image.shape]
            out[coords] = 1

            # Pepper mode
            num_pepper = np.ceil(self.salt_amount * image.size * (1. - s_vs_p))
            coords = [np.random.randint(0, i - 1, int(num_pepper))
                      for i in image.shape]
            out[coords] = 0
            return out
        elif noise_typ == "poisson":
            vals = len(np.unique(image))
            vals = 2 ** np.ceil(np.log2(vals))
            noisy = np.random.poisson(image * vals) / float(vals)
            return noisy
        elif noise_typ == "speckle":
            row, col = image.shape
            gauss = np.random.randn(row, col)
            gauss = gauss.reshape(row, col)
            noisy = image + image * gauss
            return noisy



    def make_char_ims(self, font_file):
        """ get characters as numpy arrays"""

        font_size = self.output_height * 4

        font = ImageFont.truetype(font_file, font_size)

        height = max(font.getsize(c)[1] for c in self.chars)
        width =  max(font.getsize(c)[0] for c in self.chars)
        for c in self.chars:

            im = Image.new("RGBA", (width, height), (0, 0, 0))

            draw = ImageDraw.Draw(im)
            draw.text((0, 0), c, (255, 255, 255), font=font)
            scale = float(self.output_height) / height
            im = im.resize((int(width * scale), self.output_height), Image.ANTIALIAS)
            not_moved = np.array(im)[:, :, 0].astype(np.float32) / 255.
            minx,maxx = self.getMinAndMaxX(not_moved)
            cmx=np.average([minx,maxx])
            miny,maxy = self.getMinAndMaxY(not_moved)
            cmy=np.average([miny,maxy])

            cm = ndimage.measurements.center_of_mass(not_moved)
            rows,cols = not_moved.shape
            dy = rows/2 - cmy
            dx = cols/2 - cmx
            M = np.float32([[1,0,dx],[0,1,dy]])
            dst = cv2.warpAffine(not_moved,M,(cols,rows))
            yield c, dst

    def rotate(self, image):
        cols=image.shape[1]
        rows= image.shape[0]
        halfcols=cols/2
        average_color = 0.5*(np.average(image[0][:]) + np.average(image[rows-1][:]))
        print("COLOR",average_color)
        M = cv2.getRotationMatrix2D((cols/2,rows/2),self.angle,1)
        return cv2.warpAffine(image,M,(cols,rows),borderValue=average_color)


    def generate_positives_for_haarcascade(self, font_file=None, repeat=20, positive_dir='Positives'):
        """ generate positive training samples containing characters that are
        noised and rotated"""
        import os
        font_char_ims = dict(self.make_char_ims(font_file=font_file))
        random.seed()
        if not os.path.exists(positive_dir):
            os.makedirs(positive_dir)

        for condition in range(repeat):
            myrandoms = np.random.random(4)
            print("myrandoms ", myrandoms)
            self.angle = random.gauss(0, 15)
            for mychar, img in font_char_ims.items():
                myones = np.ones(img.shape)
                if myrandoms[0] < 0.5:
                    img = self.noisy("poisson",img )
                if myrandoms[1] < 0.5:
                    self.sigma = max(0.01, np.random.normal(0.2, 0.1, 1))
                    img = self.noisy("gauss",img )
                if myrandoms[2] < 0.3:
                    self.salt_amount = max(0.01, np.random.normal(0.05, 0.05, 1))
                    img = self.noisy("sp",img )
                if myrandoms[3] < 0.5:
                    img = self.noisy("speckle",img )
                img = self.rotate(image=img)
                img=255*(myones-img)
                cv2.imwrite(positive_dir+'/'+mychar+str(condition)+'.tif', img)
                #plt.imshow(img)
                #plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
                #plt.show()

    def generate_ideal(self, font_file=None, positive_dir='PositivesIdeal'):
        """ write characters once without distorsions"""
        import os
        font_char_ims = dict(self.make_char_ims(font_file=font_file))
        if not os.path.exists(positive_dir):
            os.makedirs(positive_dir)
        for mychar, img in font_char_ims.items():
            myones = np.ones(img.shape)
            img=255*(myones-img)
            cv2.imwrite(positive_dir+'/'+mychar+'.tif', img)



if __name__ == '__main__':
    import sys, glob
    from matplotlib import pyplot as plt

    font_file = sys.argv[1]
    app1 = GetCharatersForTraining()
    #app1.generate_ideal(font_file=font_file)

    #sys.exit()
    app1.generate_positives_for_haarcascade(font_file=font_file, repeat=40)


    font_char_ims = dict(app1.make_char_ims(font_file=font_file))
    for mychar, img in font_char_ims.items():
        print ("mychar: ",mychar, img.shape )
        img_noisy = app1.noisy("speckle", img)
        img_rotated = app1.rotate(image=img_noisy, angle=-10)
        plt.imshow(img_rotated)
        plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        plt.show()
