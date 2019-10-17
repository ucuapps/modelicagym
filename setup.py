from setuptools import setup

setup(
    name='modelicagym',
    version='1.0',
    packages=['modelicagym.examples', 'modelicagym.environment', 'gymalgs.rl'],
    package_dir = {
        'modelicagym.examples': './examples'
    }
    url='',
    license='',
    author='Oleh',
    author_email='',
    description='Library for Reinforcement Learning application to Modelica models using OpenAI Gym'
)
