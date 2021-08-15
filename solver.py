from Models.Category_DeepNetwork import Classification_Network,CCML_Network
import time
from Tools.utils import save_opt,makedirs


def Selecting_Framework(name="Classification_Network"):
    print("selecting %s training framework!!"%name)
    if name == "Classification_Network":
        return Classification_Network
    if name == "CCML_Network":
        return CCML_Network


def trainer(args):

    model_name = "best_model"
    logfile = "_log"
    checkpoint_path = './checkpoint/' + args.filename
    # checkpoint path
    makedirs(checkpoint_path)

    if (args.debug == True):
        args.epochs = 1

    save_opt(args)

    Model = Selecting_Framework(name=args.framework)
    train_model = Model(args,checkpoint_path = checkpoint_path)

    max_val_acc = 0
    max_top5_val_acc = 0

    start_time = time.time()
    with open(checkpoint_path + "/"+model_name+logfile+".txt", "w") as f:
        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = train_model.train(epoch)
            val_acc,val_top5_acc = train_model.validation()

            print('Val set accuracy: {0:.2f}%, Best accuracy: {1:.2f}%'.format(val_acc, max_val_acc))
            print('Val set Top 5  accuracy: {0:.2f}%, Best accuracy: {1:.2f}%'.format(val_top5_acc, max_top5_val_acc))

            f.write("[Epoch {0:3d}] Val set accuracy: {1:.3f}%, , Best accuracy: {2:.2f}% \n".format(epoch, val_acc, max_val_acc))
            f.write("[Epoch {0:3d}] Val set Top5 accuracy: {1:.3f}%, , Best accuracy: {2:.2f}% \n".format(epoch, val_top5_acc, max_top5_val_acc))

            if max_val_acc < val_acc:
                max_val_acc = val_acc
                train_model.save_network(epoch,val_acc,file_name=checkpoint_path + '/' + model_name + '_ckpt.t7')

            if max_top5_val_acc < val_top5_acc:
                max_top5_val_acc = val_top5_acc

        time_interval = time.time() - start_time
        time_split = time.gmtime(time_interval)
        print("Training time: ", time_interval, "Hour: ", time_split.tm_hour, "Minute: ", time_split.tm_min, "Second: ", time_split.tm_sec)
        f.write("Training time: " + str(time_interval) + "Hour: " + str(time_split.tm_hour) + "Minute: " + str(
            time_split.tm_min) + "Second: " + str(time_split.tm_sec))
        f.write("\n")

    #get the best model
    train_model.load_network(path=checkpoint_path,name="best_model_ckpt.t7")

    #testing
    train_model.test(Vis_files=True)
    #testing speed
    train_model.test_speed()
    #visualization for CCML
    train_model.visualization()
    #
    train_model.calculate_IoU()


def tester(args):

    checkpoint_path = './checkpoint/' + args.filename
    # checkpoint path
    makedirs(checkpoint_path)

    if (args.debug == True):
        args.epochs = 1

    save_opt(args)

    Model = Selecting_Framework(name=args.framework)
    train_model = Model(args, checkpoint_path=checkpoint_path)
    train_model.load_network(path=checkpoint_path,name="best_model_ckpt.t7")

    # testing
    train_model.test()
    # testing speed
    train_model.test_speed()
    # visualization for CCML
    train_model.visualization()
    #
    train_model.calculate_IoU()





