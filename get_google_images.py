# python3 "google search terms"
# this writes pictures do dir "GOOGLE_SEARCH_TERMS"

from bs4 import BeautifulSoup
import requests
import re
import os
import sys
from urllib.request import urlretrieve

try:
    import cookielib
except ImportError:
    import http.cookiejar
    
try:
    import urllib.request as urllib2
except ImportError:
    import urllib2
try:
    import json
except ImportError:
    # python 2.5
    import simplejson as json    

def get_soup(url,header):
    return BeautifulSoup(urllib2.urlopen(urllib2.Request(url,headers=header)),'html.parser')


#query = input("query image: ")# you can change the query for the image  here
image_type="ActiOn"
query = sys.argv[1]
query= query.split()
query='+'.join(query)
url="https://www.google.co.in/search?q="+query+"&source=lnms&tbm=isch"
print(url)
#add the directory for your image here
header={'User-Agent':"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/43.0.2357.134 Safari/537.36"
}
soup = get_soup(url,header)


ActualImages=[]# contains the link for Large original images, type of  image
for a in soup.find_all("div",{"class":"rg_meta"}):
    link , Type =json.loads(a.text)["ou"]  ,json.loads(a.text)["ity"]
    ActualImages.append((link,Type))

print("there are total" , len(ActualImages),"images")

###save images
newdir = query.upper().replace(' ','_')
if not os.path.exists(newdir):
    os.makedirs(newdir)

# try to get more than 100 results
#from selenium import webdriver    
#browser = webdriver.Opera()
#browser.get(url)
#for _ in range(500):
#    browser.execute_script("window.scrollBy(0,10000)")    

for i , (img , Type) in enumerate( ActualImages):
    #try:
    filename = img.split("/")[-1]
    print("img:", img)
    print("filename:", filename)
    if ((filename.endswith('jpg') or (filename.endswith('png')))):
        try:
            urlretrieve(img, newdir+'/'+filename)
        except:
            print("skipping:", img)
    else:
        print("skipping:", filename)
