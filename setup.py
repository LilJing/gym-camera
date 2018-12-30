from setuptools import setup
import sys


setup(name='gym_camera',
      install_requires=['gym', 
                        'numpy', 
                        'matplotlib', 
                        'scikit-image', 
                        'jupyterlab'],
      description='gym-camera: A 2D camera gym environment ',
      author='Xingdong Zuo',
      url='https://github.com/zuoxingdong/gym-maze',
      version='0.1'
)
