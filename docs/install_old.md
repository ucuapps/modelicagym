## Installation instructions
These instructions can be used to setup environment with specific versions of libraries.
To this end, corresponding versions should be specified instead of used further.

### Test setup description
v1.0 of the toolbox was tested on the following setup:

* Ubuntu 18 64 Bit machine
* Python 3.6.8
* Java 8
* Assimulo 2.9
* PyFMI 2.3.1
* Sundials 2.4.0
* Ipopt 3.12.12
* *(optional)* Dymola 2017 - if one wants to to compile own FMU.
If one wants to use an FMU exported from Dymola for simulation, licence file should be available.
* *(optional)* JModelica 2.4 - if one wants to compile own FMU.

### Python and Python modules
Backward compatibility is expected, but not guaranteed.
Use Python 3.6, as it was utilized in the test setup.

#### Install Python and other necessary packages
	sudo apt-get update	
	sudo apt-get install python3-pip python3-dev
	sudo apt-get install libatlas-base-dev gfortran libffi-dev
	sudo apt-get install libfreetype6-dev
	sudo apt-get install cmake

You can check Python version with the following command:

    python3 -V
    
#### Install modules
	pip3 install --upgrade numpy
    pip3 install --upgrade scipy
    pip3 install --upgrade nose
    pip3 install --upgrade pandas
    pip3 install --upgrade matplotlib
    pip3 install --upgrade sympy	
    pip3 install --upgrade jupyter
    pip3 install --upgrade pytest
    pip3 install --upgrade Cython
    pip3 install --upgrade lxml
### Install PyFMI
Following steps are executed from your user home directory. 
If starting from other point, please adjust paths respectively.

####  Create working dir
	mkdir pyfmi

#### Install FMI Library
	cd pyfmi
    sudo wget http://www.jmodelica.org/fmil/FMILibrary-2.0.3-src.zip
    sudo unzip FMILibrary-2.0.3-src.zip
    cd FMILibrary-2.0.3/
    sudo mkdir build-fmilib && cd build-fmilib
    sudo cmake -DFMILIB_INSTALL_PREFIX=../install ../
    sudo make install test

#### Install Sundials
	cd pyfmi
	sudo wget http://computation.llnl.gov/projects/sundials-suite-nonlinear-differential-algebraic-equation-solvers/download/sundials-2.4.0.tar.gz
	tar xzf sundials-2.4.0.tar.gz
	cd sundials-2.4.0
	sudo ./configure CFLAGS="-fPIC"
	sudo make install
	
#### Install Assimulo
	cd pyfmi	
	sudo wget https://pypi.python.org/packages/4c/c0/19a54949817204313efff9f83f1e4a247edebed0a1cc5a317a95d3f374ae/Assimulo-2.9.zip
 	sudo unzip Assimulo-2.9.zip
	cd Assimulo-2.9
    sudo python3 setup.py install --sundials-home=/usr/local/ --blas-home=/usr/lib/x86_64-linux-gnu/ --lapack-home=/usr/lib/x86_64-linux-gnu/ --log=DEBUG

#### Install PyFMI
	cd pyfmi
	sudo wget https://pypi.python.org/packages/66/60/26664b2b2cad4a7fae409214e2f8901177322d78bfb11ef61e580115c9b8/PyFMI-2.3.1.zip#md5=577829ee1ee83fbb8c28ddf4b82aa4ee
	sudo unzip PyFMI-2.3.1.zip
	cd PyFMI-2.3.1
	sudo python3 setup.py install --fmil-home=/home/<your_username>/pyfmi/FMILibrary-2.0.2b3/install/

You can try installing latest version using pip. 
However this version & setup was not tested:
    
    pip3 install pyfmi


#####  Sources: 
* [FMILibrary. Official docs](https://jmodelica.org/fmil/FMILibrary-2.0.3-htmldoc/index.html)
* [Sundials. Official Installation guide](https://github.com/LLNL/sundials/blob/master/INSTALL_GUIDE.pdf)
* [Assimulo. Official Installation guide](https://jmodelica.org/assimulo/installation.html)
* [PyFMI. Official Installation guide](https://jmodelica.org/pyfmi/installation.html)
* [PyFMI on pypi](https://pypi.org/project/PyFMI/)

### Install OpenAI gym with libav-tools for visualization purposes

First:

    sudo apt-get install libav-tools
    
Install with pip:    

    pip install gym

**Or** you can build it from sources:

	git clone https://github.com/openai/gym.git
	cd gym
	pip install -e .

Finally:

	sudo pip3 install pyglet

#####  Source: 
*  [Official Documentation](http://gym.openai.com/docs/#installation)


###  Install Java 8
    sudo apt install openjdk-8-jdk
If you are using Oracle distribution of Java 8, this should be fine as well.

## Optional: 

### Dev-tools
You may want to install PyCharm and Jupyter Notebooks for convenience. 

#### PyCharm
Simply execute:

    sudo apt install snapd snapd-xdg-open
    sudo snap install pycharm-community --classic

Run PyCharm from the list of applications.

#### Jupyter Notebook
You can install it with the following command:

    python3 -m pip install --upgrade pip
    python3 -m pip install jupyter

Use `jupyter notebook` command to run it. 


### Modelica tools 
	
As it was mentioned, you need Dymola or JModelica, if you are going to recompile FMU's.

### JModelica
First, install all required packages:

    sudo apt-get -y install g++
    sudo apt-get -y install subversion
    sudo apt-get -y install gfortran
    sudo apt-get -y install ipython
    sudo apt-get -y install cmake
    sudo apt-get -y install swig
    sudo apt-get -y install ant
    sudo apt-get -y install openjdk-8-jdk
    sudo apt-get -y install python-dev
    sudo apt-get -y install python-numpy
    sudo apt-get -y install python-scipy
    sudo apt-get -y install python-matplotlib
    sudo apt-get -y install cython
    sudo apt-get -y install python-lxml
    sudo apt-get -y install python-nose
    sudo apt-get -y install python-jpype
    sudo apt-get -y install zlib1g-dev
    sudo apt-get -y install libboost-dev

#### Install IPopt
Download [Ipopt](https://www.coin-or.org/download/source/Ipopt/) and run following commands:

    tar xvf Ipopt-<version>.tgz
    cd Ipopt-<version>/ThirdParty/Blas
    ./get.Blas
    cd ../Lapack
    ./get.Lapack
    cd ../Mumps
    ./get.Mumps
    cd ../Metis
    ./get.Metis
    cd ../../
    
    mkdir build
    cd build
    ../configure --prefix=/home/<user_name>/<ipopt_installation_location>
    make install

#### JModelica
First, checkout it using subversion:
    
    svn co https://svn.jmodelica.org/trunk JModelica

Then run following commands:

    cd JModelica
    mkdir build
    cd build
    ../configure --prefix=/home/<user_name>/<jmodelica_install_location> \
     --with-ipopt=/home/<user_name>/<ipopt_install_location>
    make install
    
### Dymola

Dymola for Linux is distributed as a tar file that contains rpm files. 
Replace the first line with the actual Dymola version and run the following commands:

    # Change DYMVER to the version you are installing
    DYMVER=dymola-2016-x86_64
    # Unzip or untar the file, for example if there is a file ${DYMVER}-linux.tar.gz, type
    tar xzf ${DYMVER}-linux.tar.gz
    # Unpack the rpm files
    cd ${DYMVER}-linux
    for ff in `ls *.rpm`; do (sudo rpm2cpio $ff | cpio -i --make-directories); done
    sudo chmod -R o-w opt/${DYMVER}
    sudo mv opt/${DYMVER} /opt/${DYMVER}

    # Dymola is now installed in /opt/$DYMVER  
    # We will now make it executable with a simple dymola command
    sudo ln -s /opt/${DYMVER}/bin/dymola.sh /usr/local/bin/dymola

    # Now move the man page to the right location (create the folder first if you get an error)
    sudo mv usr/local/man/man1/dymola.1 /usr/local/share/man/man1/dymola.1

	cd /opt
	sudo unzip DYMOLA_2017_AllLinux.zip -d DYMOLA_2017
	cd DYMOLA_2017/linux_x86_64/
	sudo alien -i -k dymola-2017.1-1.x86_64.rpm

#### Source
These instructions were taken from [here](https://simulationresearch.lbl.gov/modelica/installDymolaLinux.html).
