from setuptools import setup

setup(
    name='modelicagym',
    version='1.0',
    packages=['modelicagym.environment', 'modelicagym.gymalgs.rl'],
    package_dir={
        'modelicagym.gymalgs.rl': './gymalgs/rl'
    },
    url='github.com/ucuapps/modelicagym',
    license='GPL-3.0',
    author='Oleh Lukianykhin',
    author_email='lukianykhin@ucu.edu.ua',
    description='Library for Reinforcement Learning application to Modelica models using OpenAI Gym'
)
