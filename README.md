**Update** Recently I am building my own deep learning righ and I have my first NVIDIA 1080Ti GPU. I refered [Slav's blog](https://blog.slavv.com/the-1700-great-deep-learning-box-assembly-setup-and-benchmarks-148c5ebe6415) and [Waydegg's blog](http://forums.fast.ai/t/a-guide-for-creating-a-server-for-deep-learning-assembly-server-library-setup/19082) to build my machine.
**Note** After revised this blog I have found many other blogs or tutorials on this:
**CUDA Official Document** [here](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#post-installation-actions)
**CUDNN official Document** [here](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html#install-linux)
1. useful in the removing old kernels[this](http://christopher5106.github.io/nvidia/2016/12/30/commands-nvidia-install-ubuntu-16-04.html)
2. trouble shooting part is useful [this](http://queirozf.com/entries/installing-cuda-tk-and-tensorflow-on-a-clean-ubuntu-16-04-install#the-distribution-provided-pre-install-script-failed-are-you-sure-you-want-to-continue)
3. Dell XPS is somewhat like my Inspiron 15 7000 laptop [this](https://gist.github.com/whizzzkid/37c0d365f1c7aa555885d102ec61c048)
4. Read Carefully from the official blog[this](https://devtalk.nvidia.com/default/topic/1023490/suffering-on-cuda-amp-nvidia-driver-installation-on-ubuntu-16-04-3/)
5. **Most important and promissing one** [here](https://www.pugetsystems.com/labs/hpc/The-Best-Way-To-Install-Ubuntu-16-04-with-NVIDIA-Drivers-and-CUDA-1097/)
6. **Linuxidc blog** [this](www.linuxidc.com/Linux/2017-01/139319.htm)
7. **Nvidia Blog**[this](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#runfile)
8. **If using Optimus**[this](https://www.pcsuggest.com/nvidia-optimus-ubuntu/)
9. **Somewhat useful**[this](https://www.cnblogs.com/sp-li/p/7680526.html)
10. **-no-opengl-files** when install cuda [this](https://blog.csdn.net/ghw15221836342/article/details/79571559)
11. [this](https://zhuanlan.zhihu.com/p/25193943)
12. **verbose** [don't know whether is useful](https://zhuanlan.zhihu.com/p/27794010)
13. **seems useful**[this](https://zhuanlan.zhihu.com/p/35670162)
14. **maybe useful for bumblebee**[this](https://www.linuxidc.com/Linux/2017-02/140910.htm)
15. **MOre** [this](https://github.com/mananpal1997/CUDA-Instructions)
16. **Maybe** [this](https://github.com/sonic1sonic/Installation-Guide-for-NVIDIA-Driver-and-CUDA)


---
**Clarify about the gcc version** the official document says 5.3.1 is needed for ubuntu 16.04 however the 5.4 is installed 
it not true. 

**How to run text level 3*

8
down vote
Instead of text use runlevel 3:

GRUB_CMDLINE_LINUX="3"

# To remove all the fancy graphics you need to get rid of `splash`.
GRUB_CMDLINE_LINUX_DEFAULT=”quiet”

# Uncomment to disable graphical terminal (grub-pc only) 
GRUB_TERMINAL=console
Then update-grub and reboot.

But you really only need GRUB_CMDLINE_LINUX="3". For quick test hit ESC during booting to get into the grub boot menu. Then press e and find the line which specifies kernel and add 3 at the end:

 linux /vmlinuz root=/dev/mapper/ubuntu ro 3
Boot it with CTRL+x

Ideally I also want to be able to start GUI by typig a command.

One of these:

$ sudo telinit 5
$ sudo service lightdm restart
$ sudo systemctl start lightdm
Tested on Ubuntu 16.04.1 LTS.
---
# Nvidia CUDA and Gym(Dart, Pybullet planned) install on ubuntu 16.04    
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
---
**My Way to install the driver**
1. download the NVIDIA-Linux 390 version driver
2. logout the desktop and CTRL+ALT+Fn4 execute:
```bash
# I don't know which is helpful and excute both
sudo sudo service lightdm stop
sudo systemctl stop lightdm
```
3. run the driver with --no-opengl-files **this is important**
4. reboot 
5. In order not to let the installation of cuda destroy the desktop, I logout and do (2) to install cuda, remember to add PATH variable and ldconfig as like [here](https://gist.github.com/Kelvinson/b9a6124ec1617d52e86ae8975e118277#install-cuda)
```bash
sudo bash -c "echo /usr/local/cuda/lib64/ > /etc/ld.so.conf.d/cuda.conf"
sudo ldconfig
```
6. Then I reboot to see whether I can log into the desktop, It's OK! Then I installed the 3 updates on cuda-9.1
7. The I have to install cudnn
```bash
# do this as WilliamFalcon's blog 
wget https://s3.amazonaws.com/open-source-william-falcon/cudnn-9.0-linux-x64-v7.1.tgz  
sudo tar -xzvf cudnn-9.0-linux-x64-v7.1.tgz  
# the -P flag is important becuase if not use -P, error "libcudnn.so.7 is not a valid link" will occur
sudo cp -P  cuda/include/cudnn.h /usr/local/cuda/include # -p retains the link
sudo cp -P cuda/lib64/libcudnn* /usr/local/cuda/lib64 
sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*
# Add these lines to end of ~/.bashrc:
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64"
export CUDA_HOME=/usr/local/cuda
# Reload bashrc
source ~/.bashrc
```
8. also remember to ldconfig
```
sudo ldconfig
```
9. Remeber to do the post installation things as [here](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#post-installation-actions) includeing persistance
```bash
/usr/bin/nvidia-persistenced --verbose
```

Note: 
1. Because at first I do not copy with the -P flag, there comes with the link error. 
2. Using the above way to install cudnn, there is no cudnn examples to verify the cudnn installation, I tried to install the debian verson of cudnn exmples but comes the error "dependencies unmet", it requires install deb version of cudnn first. So I have to manully uninstall the cudnn: sudo apt-get autoremove cudnn_7-1.docs.
3. In this part I leant a lot from [gist](https://gist.github.com/Kelvinson/b9a6124ec1617d52e86ae8975e118277#install-cuda) from wangruohui 
4. After that I tested by installing GPU pytorch and other scripts, all seems right. But I found that accidently the Nvidia X server settings pops up and gives the error "You don't appear to be using the Nvidia X server. Please edit your configuration file (just run 'nvidia-xconfig') and restart the X server. 

I can only to do it and the /etc/X11/xorg.conf file is created. Now i have to risk to restart to see what will happen.(lol)
---
**More about Ubuntu**
1. Keep the external monitor turn on when close the lid of the laptop:
```bash
# /etc/UPower/UPower.conf, and change ignoreLid to true.
ignoredLid=true
```
---

3. Install cudnn   
``` bash
wget https://s3.amazonaws.com/open-source-william-falcon/cudnn-9.0-linux-x64-v7.1.tgz  
sudo tar -xzvf cudnn-9.0-linux-x64-v7.1.tgz  
sudo cp -P cuda/include/cudnn.h /usr/local/cuda/include
sudo cp -P cuda/lib64/libcudnn* /usr/local/cuda/lib64
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

- optional: Install tensorflow with GPU support for python 3.5
``` bash
pip install tensorflow-gpu

# If the above fails, try the part below
# pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.2.0-cp36-cp36m-linux_x86_64.whl

```   
** However, becasue I install the the 9.1/7.1 cuda/cudnn, because the pip and the wheel file is prebuilt with cuda 9.0 and cudnn 7.1 I have to install tensorflow from building source. In the "./configure" sesssion, select 'no' if you are not familiar with those options, like 'MPI' 'TensorRT'. I select them at first and bazel build fail. I then bazel clean and reconfigure it with the most simple configuration and passed. 
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
# other tools to install: kdictionary
```
sudo snap install kdictionary
```

10. You can choose to install Dart and Dartsim or optionally Gazebo to enrich your simulation tools. However, I doubt there is a bug about OpenGL rendering conflicts with Nvidia drivers that cause my machine to crash(cannot login in again after resstart) So I choose not to install them temporailly.

11. More about Simulators
a. About the benchmarking of the 4 simulators(Dart, Bullet, ODE) see here[]()
b. Use cases mentioned by Pybullet: [Simulation to domain Adaptation](https://sites.google.com/view/graspgan) [Neural Task Programming:Learning to Generalize Across Hierarchical Tasks](https://stanfordvl.github.io/ntp/?utm_content=buffer8b1fc) [OpenAI Roboschool](https://blog.openai.com/roboschool/)
