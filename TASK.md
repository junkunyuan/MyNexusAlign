1. 将validation、evaluation从training的pipeline中解耦出来，单独跑eval.sh进行evaluation，validation先暂时丢弃。
2. 当前的训练pipeline并没有调用Trainer，请使用Trainer来统一训练，调用trainer.run()开启正式的训练。