"""
Author: Steve Paul 
Date: 3/1/22 """

from tensorboard import program

# tb = program.TensorBoard(default.PLUGIN_LOADERS, default.get_assets_zip_provider())
# tb.configure(argv=['--logdir', "PPO_1"])
# tb.main()

tracking_address = "logger/Trained" # the path of your log file.

if __name__ == "__main__":
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', tracking_address])
    tb.main()
    # url = tb.launch()
    # print(f"Tensorflow listening on {url}
