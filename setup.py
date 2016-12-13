from setuptools import setup, find_packages

setup(
    name='dnncode',
    version='1.0.0',
    author='Hassan A. Kingravi',
    author_email='hkingravi@gmail.com',
    description='Personal implementation of DNNs',
    packages=find_packages(exclude=['*.test', 'test']),
    install_requires=[]
)
