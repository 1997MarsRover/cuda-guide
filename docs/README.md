## Overview of Cuda Kernel excecution

When a kernel is invoked, or launched, it is executed as grid of parallel
threads. In , the launch of Kernel 1 creates Grid 1. Each CUDA
thread grid typically is comprised of thousands to millions of lightweight 
GPU threads per kernel invocation. Creating enough threads to fully utilize
the hardware often requires a large amount of data parallelism; for example,
each element of a large array might be computed in a separate thread.

<div>
  <img src="assets/image.png" alt="thread">
</div>

Threads in a grid are organized into a two-level hierarchy, as illustrated
in Figure 3.13. For simplicity, a small number of threads are shown in
Figure 3.13. In reality, a grid will typically consist of many more threads.
At the top level, each grid consists of one or more thread blocks. All blocks
in a grid have the same number of threads. In Figure 3.13, Grid 1 is
organized as a 22 array of 4 blocks. Each block has a unique twodimensional coordinate given by the CUDA specific keywords blockIdx.x
and blockIdx.y. All thread blocks must have the same number of threads
organized in the same manner.