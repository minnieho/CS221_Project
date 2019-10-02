from setuptools import setup, find_packages

setup(name='gym_act',
    version='1.0.0',
    description='An environment for Anti Collision Tests in Autonomous Driving',
    url='https://github.com/PhilippeW83440/CS221_Project/gym-act',
    author='Philippe Weingertner',
    author_email='philippe.weingertner@gmail.com',
    license='MIT',
)

keywords='autonomous driving simulation environment reinforcement learning'
install_requires=['gym','numpy', 'opencv-python', 'jupyter']
