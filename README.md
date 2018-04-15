# Tensorflow GPU install on ubuntu 16.04    

These instructions are intended to set up a deep learning environment for GPU-powered tensorflow.      
[See here for pytorch GPU install instructions](https://github.com/williamFalcon/pytorch-gpu-install)

After following these instructions you'll have:

1. Ubuntu 16.04. 
2. Cuda 9.0 drivers installed.
3. A conda environment with python 3.6.    
4. The gym environment with Mujoco and Robotics fully integrated   
---
### Installing OS Before it starts
Ususally I will encounter kinds of GRUB problems and I choose to reinstall the Windows and Ubuntu totally.
1. first I use windows installer disk to reinstall windows, I can choose to delete the two disks (windows , ubuntu) when I choose where to install windows.
2. After I install windows, write the 16.04.3 image(16.04.3 works better for me than 16.04.4) to disk and select the boot sequence from the disk and install ubuntu, on the partitioning I refer [here](https://jingyan.baidu.com/article/fb48e8be1480486e622e14ed.html)-that is 200M/8G/50G/the remaining part
3. sudo apt-get autoremove linux-image-多余的内核版本 (to avoid future errors of gzip: no space left on device)

---   
### Step 0: Noveau drivers     
Before you begin, you may need to disable the opensource ubuntu NVIDIA driver called [nouveau](https://nouveau.freedesktop.org/wiki/).

**Option 1: Modify modprobe file**
1. After you boot the linux system and are sitting at a login prompt, press ctrl+alt+F1 to get to a terminal screen.  Login via this terminal screen.
2. Create a file: /etc/modprobe.d/nouveau
3.  Put the following in the above file...
```
blacklist nouveau
options nouveau modeset=0
```
4. reboot system   
```bash
reboot
```   
    
5. On reboot, verify that noveau drivers are not loaded   
```
lsmod | grep nouveau
```

If `nouveau` driver(s) are still loaded do not proceed with the installation guide and troubleshoot why it's still loaded.    

**Option 2: Modify Grub load command**    
From [this stackoverflow solution](https://askubuntu.com/questions/697389/blank-screen-ubuntu-15-04-update-with-nvidia-driver-nomodeset-does-not-work)    

1. When the GRUB boot menu appears : Highlight the Ubuntu menu entry and press the E key.
Add the nouveau.modeset=0 parameter to the end of the linux line ... Then press F10 to boot.   
2. When login page appears press [ctrl + ALt + F1]    
3. Enter username + password   
4. Uninstall every NVIDIA related software:   
```bash    
sudo apt-get purge nvidia*  
sudo reboot   
```   
**my option**
Neither option 1 or options2 works for me, I refer the Nvidia docs and do the following:
0. After you boot the linux system and are sitting at a login prompt, press ctrl+alt+F4(not CTRL+ALT+F1) to get to a terminal screen.  Login via this terminal screen.
1.	Create a file at /etc/modprobe.d/blacklist-nouveau.conf with the following contents:
    ```
    blacklist nouveau
	options nouveau modeset=0
    ```
3.	Regenerate the kernel initramfs:$ sudo update-initramfs -u
4. reboot  
5. to see whether nouveau is blacklisted
    ```
    lsmod | grep nouveau
    ```
---   
## Installation steps     


0. update apt-get   
``` bash 
sudo apt-get update
```

1. Installing kinds of building essentials
- Install apt-get deps  
``` bash
sudo apt-get install openjdk-8-jdk git python-dev python3-dev python-numpy python3-numpy build-essential python-pip python3-pip python-virtualenv swig python-wheel libcurl3-dev   
```
- Install essentials for gym environment:
```bash
sudo apt-get install -y python-numpy python-dev cmake zlib1g-dev libjpeg-dev xvfb libav-tools xorg-dev python-opengl libboost-all-dev libsdl2-dev swig
```
2. install nvidia drivers 
``` bash
# The 16.04 installer works with 16.10.
# download drivers
curl -O http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_9.0.176-1_amd64.deb

# download key to allow installation
sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub

# install actual package
sudo dpkg -i ./cuda-repo-ubuntu1604_9.0.176-1_amd64.deb

#  install cuda (but it'll prompt to install other deps, so we try to install twice with a dep update in between
sudo apt-get update
sudo apt-get install cuda-9-0   
```    

2a. reboot Ubuntu
```bash
sudo reboot
```    

2b. check nvidia driver install 
``` bash
nvidia-smi   

# you should see a list of gpus printed    
# if not, the previous steps failed.   
``` 

3. Install cudnn   
``` bash
wget https://s3.amazonaws.com/open-source-william-falcon/cudnn-9.0-linux-x64-v7.1.tgz  
sudo tar -xzvf cudnn-9.0-linux-x64-v7.1.tgz  
sudo cp cuda/include/cudnn.h /usr/local/cuda/include
sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*
```    

4. Add these lines to end of ~/.bashrc:   
``` bash
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64"
export CUDA_HOME=/usr/local/cuda
```   

4a. Reload bashrc     
``` bash 
source ~/.bashrc
```   

5. Install miniconda   
``` bash
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh   

# press s to skip terms   

# Do you approve the license terms? [yes|no]
# yes

# Miniconda3 will now be installed into this location:
# accept the location

# Do you wish the installer to prepend the Miniconda3 install location
# to PATH in your /home/ghost/.bashrc ? [yes|no]
# yes    

```   

5a. Reload bashrc     
``` bash 
source ~/.bashrc
```   

6. Create conda env to install gym 
``` bash
conda create -n gym python=3.5.2

# press y a few times 
```   

7. Activate env   
``` bash
source activate gym
```
8. Istall 
- Install mujuco(do it as the official docs said, it is easy)
Note to Add these two lines to the end of ~/.bashrc:
```
export LD_LIBRARY_PATH="/home/kelvinson/.mujoco/mjpro150/bin:${LD_LIBRARY_PATH}"
export LD_LIBRARY_PATH="/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}"
```
- Intall the esentials of mujoco-py
 clone gym to local: inside gym env, pip install --ignore-installed pip (required by gym )
follow the [docker file](https://github.com/openai/mujoco-py/blob/master/Dockerfile)to install perquisites of mujoco-py. (I have simplified it to the file build_dependencies in the main repo) \
- now let's install the gym env:
```bash
 LD_LIBRARY_PATH=$HOME/.mujoco/mjpro150/bin pip install mujoco-py pip install -e '.[all]' --no-cache-dir
```

- After that I have to downgrade pyglet used by classic control task to 1.2.4 or the SpaceInvader cannot be rendered

- optional: Install tensorflow with GPU support for python 3.6    
``` bash
pip install tensorflow-gpu

# If the above fails, try the part below
# pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.2.0-cp36-cp36m-linux_x86_64.whl
```   
9. Test of install

9.1 Test of gym install 
Note depends on your situation, optionlly I have to downgrade the pyglet version to 1.2.4 to use the control env
```
pip uninstall pyglet
pip install -I pyglet==1.2.4
```
when run mujuco and robotics environment, I know there is a bug about omesa that has something to do with OpenGL on my ubuntu so I have to append **LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so** when I run a python program. Thus it becomes:
```
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so python YOUR_PROGRAM
```
Also when running the gym environment you have to put **env.close()**at the end of the game, or the error "NoneType" object is not iterable will occur.

```python
import gym
env = gym.make('LunarLander-v2')
for i in range(100):
    env.reset()
    env.render()
env.close()
```

9.2 Test tf install 
``` bash
# start python shell   
python

# run test script   
import tensorflow as tf   

hello = tf.constant('Hello, TensorFlow!')

# when you run sess, you should see a bunch of lines with the word gpu in them (if install worked)
# otherwise, not running on gpu
sess = tf.Session()
print(sess.run(hello))
```  
10. You can choose to install Dart and Dartsim or optionally Gazebo to enrich your simulation tools. However, I doubt there is a bug about OpenGL rendering conflicts with Nvidia drivers that cause my machine to crash(cannot login in again after resstart) So I choose not to install them temporailly.