"""
dumpimages.py
    Downloads all the images on the supplied URL, 
    saves to the current directory

Usage:
    python3 get_negative_pictures.py  http://www.mv.helsinki.fi/home/mokaukon/public_html/ [output]
"""

from bs4 import BeautifulSoup as bs
#import urllib.parse
try:
    import urllib.request as urllib2
except ImportError:
    import urllib2
from urllib.request import urlretrieve
from urllib.request import urlopen
from urllib.request import Request
try:
    from urllib.parse import urlparse, urlunparse
except ImportError:
    from urlparse import urlparse, urlunparse
#from urllib.request import urlparse
import os
import sys

def main(url, out_folder="./"):
    """Downloads all the images at 'url' to /test/"""
    #headers = { 'User-Agent' : 'Mozilla/5.0' }
    req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    soup = bs(urlopen(req))
    parsed = list(urlparse(url))

    images = [img for img in soup.findAll('img')]
    print (str(len(images)) + "images found.")
    print ('Downloading images to current working directory.')
    #compile our unicode list of image links
    image_links = [each.get('src') for each in images]
    for each in image_links:
        try:
            filename=each.split('/')[-1]
            if url.startswith('http'):
                print(each)
                urlretrieve(each, filename)
            else:
                urlretrieve(url+'/'+each, filename)
        except:
            print('skipping', each)

    
    #for image in soup.findAll("img"):
    #    print("Image: %(src)s" % image)
    #    filename = image["src"].split("/")[-1]
    #    parsed[2] = image["src"]
    #    outpath = os.path.join(out_folder, filename)
    #    print(image)
    #    if image["src"].lower().startswith("http"):
    #        urlretrieve(image["src"], outpath)
    #        print("filename1", outpath)
    #    else:
    #        print("filename2", outpath,urlunparse(parsed))
    #        #pass
    #        #urlretrieve(urlunparse(parsed), outpath)
    #        urlretrieve(image["src"], outpath)

def _usage():
    print("usage: python dumpimages.py http://example.com [outpath]")

if __name__ == "__main__":
    url = sys.argv[-1]
    #out_folder = "/test/"
    if not url.lower().startswith("http"):
        out_folder = sys.argv[-1]
        url = sys.argv[-2]
        if not url.lower().startswith("http"):
            _usage()
            sys.exit(-1)
    #main(url, out_folder)
    main(url)

    
