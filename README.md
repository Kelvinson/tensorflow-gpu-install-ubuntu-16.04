0. I install Ubuntu in the UEFI mode rather than legacy mode. It's cool!
1. Just install the cuda using the distribution-specific method, i.e download the deb file for ubuntu and go all the way. I encounter no problem this way.
2. install gym, mujoco_py. I encounter the problem of "fatal error: GL/glew.h: No such file or directory" and reinstall mujoco_py with "LD_LIBRARY_PATH=$HOME/.mujoco/mjpro150/bin pip -I install mujoco-py" didn't help. Solution of the issue [180](https://github.com/openai/mujoco-py/issues/180) helped me.
```bash
sudo apt-get install libglew-dev
```
3. Because I install CUDA 9.2 and cuDNN 7.1 however the default for tensorflow is CUDA 9.0 and cuDNN 7.0 so I have to build from source. 
- first bug is NCCL2: 

when execute "./configure" tf searches for "/usr/local/cuda/lib/libnccl.so.2" and "/usr/local/cuda/include/nccl.h" from [here](https://github.com/tensorflow/tensorflow/blob/226831aab92a395a26824a08caa9d43f0c3d604e/tensorflow/tools/ci_build/Dockerfile.gpu#L33)
.becuase CUDA 9.2 only has "/usr/local/cuda/lib64" directory but not "/usr/local/cuda/lib" so I do this:
```bash
sudo mkdir -p /usr/local/cuda/lib
sudo ln -s /usr/lib/x86_64-linux-gnu/libnccl.so.2 /usr/local/cuda/lib/libnccl.so.2
sudo ln -s /usr/include/nccl.h /usr/local/cuda/include/nccl.h
```
- second bug is cannot find libcdnn.so.7.
```bash
ERROR: /home/kelvinson/Downloads/tensorflow/tensorflow/BUILD:489:1: Linking of rule '//tensorflow:libtensorflow_framework.so' failed (Exit 1)
/usr/bin/ld: skipping incompatible bazel-out/host/bin/_solib_local/_U@local_Uconfig_Ucuda_S_Scuda_Ccudnn___Uexternal_Slocal_Uconfig_Ucuda_Scuda_Scuda_Slib/libcudnn.so.7 when searching for -l:libcudnn.so.7
/usr/bin/ld: skipping incompatible bazel-out/host/bin/_solib_local/_U@local_Uconfig_Ucuda_S_Scuda_Ccudnn___Uexternal_Slocal_Uconfig_Ucuda_Scuda_Scuda_Slib/libcudnn.so.7 when searching for -l:libcudnn.so.7
/usr/bin/ld: cannot find -l:libcudnn.so.7
collect2: error: ld returned 1 exit status
Target //tensorflow/tools/pip_package:build_pip_package failed to build
Use --verbose_failures to see the command lines of failed build steps.
INFO: Elapsed time: 2839.656s, Critical Path: 172.22s
INFO: 3492 processes: 3492 local.
FAILED: Build did NOT complete successfully
```
***whoops** Finally I found that I installed the wrong cudnn (ppc64le version Power8/Power9 for  not the regular one 64 bit) then I downloaded the right one cudnn-9.2-linux-x64-v7.1 and unzip to the cuda directory.
```bash
tar xvzf cudnn-9.2-linux-x64-v7.1.tgz
sudo cp -P cuda/include/cudnn.h /usr/local/cuda/include
sudo cp -P cuda/lib64/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*
# add the path to LD_LIBRARY_PATH
```
And finally it succeeds!
```bash
Target //tensorflow/tools/pip_package:build_pip_package up-to-date:
  bazel-bin/tensorflow/tools/pip_package/build_pip_package
INFO: Elapsed time: 4665.159s, Critical Path: 150.65s
INFO: 7837 processes: 7837 local.
INFO: Build completed successfully, 10239 total actions
```
So the safe way is to use isntall debfile for cuDNN and thus you can verify the installation by running mnistCUDNN example.
4. Besides the problems in the OLD-Readme.md, I encountered the problems below:
```bash
Creating window glfw
ERROR: GLEW initalization error: Missing GL version

Press Enter to exit ...Killed
```
The solution is preload the GLEW lib manully, add the following line to the .bashrc file :
```bash
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so:/usr/lib/nvidia-396/libGL.so
```
5. In order to setup a remote ssh access, I refered the tutorial [here](https://dev.to/zduey/how-to-set-up-an-ssh-server-on-a-home-computer) and it worked. Also to setup access remote jupyter notebook, first configure as below:
```bash
# Create a ~/.jupyter/jupyter_notebook_config.py with settings
jupyter notebook --generate-config
jupyter notebook --port=8888 --NotebookApp.token='' # Start it
```
if want the server to start jupyter on startup, we will use crontab to do this, which we can edit by running "crontab -e" . Then add the following after the last line in the crontab file:
```bash
# Replace 'path-to-jupyter' with the actual path to the jupyter
# installation (run 'which jupyter' if you don't know it). Also
# 'path-to-dir' should be the dir where your deep learning notebooks 
# would reside (I use ~/DL/).
@reboot path-to-jupyter notebook --no-browser --port=8888 --NotebookApp.token='' --notebook-dir path-to-dir &
```
then you can test it in the remote client by the following:
```bash
# Replace user@host with your server user and ip.
ssh -N -f -L localhost:8888:localhost:8888 user@host
```
