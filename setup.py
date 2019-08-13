from setuptools import setup

setup(
    name='pyxe',
    version='0.9.2',
    author='C. Simpson',
    author_email='c.a.simpson01@gmail.com',
    packages=['pyxe'],
    include_package_data=True,
    url='https://github.com/casimp/pyxe',
    download_url = 'https://github.com/casimp/pyxe/tarball/v0.9.2',
    license='LICENSE.txt',
    description='XRD strain analysis package. Efficient analysis and visulisation of diffraction data.',
    keywords = ['XRD', 'EDXD', 'x-ray', 'diffraction', 'strain', 'synchrotron'],
#    long_description=open('description').read(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows"]
)
