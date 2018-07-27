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
