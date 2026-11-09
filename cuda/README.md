# Introduction
1.cuda作为最基本的并行语言模板，应当掌握写基本算子概念，如何profile看sol。


## ncu过程
1. ncu的对象是可执行程序，在ncu之前，应当有编译的产物，使用nvcc生成可执行文件。
2. nvcc -arch=sm_89 vec_add.cu vec_add