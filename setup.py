from setuptools import setup

with open('requirements.txt') as f:
    install_reqs = f.read().splitlines()



setup(name='invivoinfer',
      version='0.1',
      description='invivoinfer, code described in the paper "Extraction of synaptic input properties in vivo",'
                  'P.Puggioni et al.',
      #url='http://github.com/storborg/funniest',
      author='Paolo Puggioni',
      author_email='p.paolo321@gmail.com',
      license='MIT',
      packages=['invivoinfer'],
      install_requires=install_reqs,
      zip_safe=False)