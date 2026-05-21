实现高性能CUDA RMSNorm，要求：

* 符合[PyTorch Pooling2D API](https://docs.pytorch.org/docs/stable/generated/torch.nn.RMSNorm.html)定义，所有Parameters都需要支持：
* 输入及输出Shape为`[N,*]`，Dim不小于2
* 输入及输出DataType为fp32、fp16、bf16
* 需要有涵盖各Parameters情况的unittests，可以直接使用PyTorch或Numpy作为golden
* 本地环境没有GPU，需要通过`ssh shuyua01@10.190.0.91`将repo拷贝到`/home/shuyua01/Development/`下进行部署、测试和调试，确保代码可以正常运行而且结果正确
* 使用`git`做版本控制
* 从baseline实现开始，实现step-by-step性能优化
* 采用经典模型（至少包括Qwen和llama，需要通过研究更多transformer模型来扩充）典型cases作为benchmark
* step-by-step性能优化和跑benchmark时，需要profiler report，分析当前性能优化点和瓶颈，每个优化步骤都要有具体的profiler metrics数据作为支撑
