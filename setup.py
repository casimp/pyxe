try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


setup(
    name='pyxe',
    version='0.8.0',
    author='C. Simpson',
    author_email='c.a.simpson01@gmail.com',
    packages=['pyxe'],
    url='https://github.com/casimp/pyxe',
    download_url = 'https://github.com/casimp/pyxe/tarball/v0.8.0',
    license='LICENSE.txt',
    description='XRD strain analysis package. Efficient analysis and visulisation of diffraction data.',
    keywords = ['XRD', 'EDXD', 'x-ray', 'diffraction', 'strain', 'synchrotron'],
#    long_description=open('description').read(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows"]
)
