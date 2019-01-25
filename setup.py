from setuptools import setup, find_packages

def _requirements():
    return [name.rstrip() for name in open('requirements.txt').readlines()]

setup(name='seq_gan',
      version='0.0.1',
      description='seq_gan with keras',
      author='Colin Gao',
      install_requires=_requirements(),
      packages=find_packages(),
      url='https://github.com/GaoQ1',
)
