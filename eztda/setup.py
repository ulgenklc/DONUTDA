import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(name='DONUTDA',
      version='1.0',
      description='Donut-like Object segmeNtation Utilizing Topological Data Analysis',
      long_description=long_description,
      long_description_content_type='text/markdown',
      url='https://github.com/ulgenklc/DONUTDA',
      author='Bengier Ulgen Kilic',
      author_email='bengieru@buffalo.edu',
      license='MIT',
      packages=setuptools.find_packages(),
      classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License"],
      install_requires=['eztda','numpy','matplotlib'],
      python_requires='>=3.6',)
