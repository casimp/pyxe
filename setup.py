try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


setup(
    name='pyxe',
    version='0.5.1',
    author='C. Simpson',
    author_email='c.a.simpson01@gmail.com',
    packages=['pyxe'],
    url='http://pypi.python.org/pypi/pyxe/',
    license='LICENSE.txt',
    description='XRD strain analysis package. Efficient analysis and visulisation of diffraction data.',
    long_description=open('README.md').read(),
)
