from distutils.core import setup
import setuptools

setup(
  name = 'oda',
  packages = ['oda'], # this must be the same as the name above
  version = '0.12',
  description = 'Analyze oceanic data',
  author = 'Zelun Wu',
  author_email = 'zelunwu@stu.xmu.edu.cn',
  # url = 'https://github.com/zelunwu/odapy',
#   keywords = ['ecco','climate','mitgcm','estimate','circulation','climate'],
  include_package_data=True,
#   data_files=[('binary_data',['binary_data/basins.data', 'binary_data/basins.meta'])],
)
