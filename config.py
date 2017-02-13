class Config:
    data_path = "./datasets/facades"
    model_path = ""
    output_path = "./results"

    img_size = 256
    img_channel = 3
    conv_channel_base = 64

    learning_rate = 0.01
    max_epoch = 100
    L1_lambda = 0.1
    save_per_epoch=1

