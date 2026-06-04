重构model部分，弃用DDP，改用FSDP。

具体来说，让model继承BaseModel，BaseModel里实现通用的FSDP wrapper。

FSDP wrapper的具体实现方法可参考：https://github.com/junkunyuan/NexusAlign/blob/master/src/nexus_align

同时ema的实现和更新也可以在BaseModel里进行统一的实现。