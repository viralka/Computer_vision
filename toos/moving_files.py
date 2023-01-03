import os

from numpy import source

source = r'D:\agnext\Agnext\OpenCv\toos\1979.txt'
destination = r'D:\agnext\Agnext\OpenCv\Videos'


os.replace(source, destination)
print('File moved ---> to source')
