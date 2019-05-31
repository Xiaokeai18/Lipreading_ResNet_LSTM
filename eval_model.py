import numpy as np
import tensorflow as tf
import pandas as pd
from model import * #frontend_3D, backend_resnet34, concat_resnet_output, fully_connected_logits
#from tfpipeline import get_batch
import input_data
#--------------------------------------------------------------------------------------------------------------------#
# USER INPUT
# dataset_dir= "/homes/mat10/Desktop/tfrecords_test/ABOUT"
# data_info_dir = "/homes/mat10/Desktop/tfrecords_test"
# data_info_dir = "/data/mat10/ISO_Lipreading/data/LRW_TFRecords"
# val_data_info = pd.read_csv(data_info_dir + "/val_data_info.csv").sample(frac=1)
# train_data_info = pd.read_csv(data_info_dir + "/train_data_info.csv").sample(frac=1)
options = {'is_training': False, 'batch_size': 8, 'num_classes': 5, 'num_epochs': 1,
           'crop_size': 80, 'horizontal_flip': False, "shuffle": False}
# specify the model directory
# srcdir = "/data/mat10/ISO_Lipreading/models/saved_models_full"
modeldir = "./saved_models_full"
model = "model_full_epoch"
savedir = "./saved_models_full/"
#--------------------------------------------------------------------------------------------------------------------#

# print("Total number of train data: %d" % data_info.shape[0])
test_list_file = 'test.list'
num_test_videos = len(list(open(test_list_file,'r')))

print("Total number of train data: %d" % num_test_videos)
number_of_steps_per_epoch = num_test_videos // options['batch_size']
# print("Total number of steps: %d" % number_of_steps)

# modelid = 17
# data_info = val_data_info

images_placeholder = tf.placeholder(tf.float32, shape=(options['batch_size'],24,options['crop_size']-20,options['crop_size'],3))
labels_placeholder = tf.placeholder(tf.int64, shape=(options['batch_size']))

prediction = frontend_3D(images_placeholder)
prediction = backend_resnet34(prediction)
prediction = blstm_2layer(prediction)
prediction = fully_connected_logits(prediction, options['num_classes'])
norm_score = tf.nn.softmax(prediction)

#cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels_placeholder, logits=prediction))

predicted_class = tf.argmax(prediction, axis=1)
correct_prediction = tf.equal(labels_placeholder, predicted_class)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

config = tf.ConfigProto(allow_soft_placement = True)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

for modelid in reversed(range(6, 10)):

    print("Evaluating model %d" % modelid)
    model = savedir + "model_full_epoch" + str(modelid)

    sess.run(tf.global_variables_initializer())


    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()
    saver.restore(sess, model)
    print("Model restored.")
    write_file = open("predict_ret.txt", "w+")
    # metrics_ = []
    # labels_ = []
    # predicted_ = []
    acc_cnt,acc5_cnt,cnt = 0,0,1
    next_start_pos = 0
    for step in range(number_of_steps_per_epoch):
        test_images, test_labels, next_start_pos, _, valid_len = input_data.read_clip_and_label(
                                filename=test_list_file,
                                batch_size=options['batch_size'],
                                num_frames_per_clip=24,
                                crop_size=options['crop_size'],
                                start_pos=next_start_pos,
                                shuffle=True
                                )
        
        predict_score = norm_score.eval(
            session=sess,
            feed_dict={images_placeholder: test_images}
            )
        acc5 = tf.nn.in_top_k(predict_score,test_labels,5)
        top5_score = acc5.eval(session=sess,
                    feed_dict={images_placeholder: test_images}
                    )
        for i in range(0, valid_len):
            true_label = test_labels[i]
            top1_predicted_label = np.argmax(predict_score[i])
            # Write results: true label, class prob for true label, predicted label, class prob for predicted label
            write_file.write('{}, {}, {}, {}\n'.format(
                    true_label,
                    predict_score[i][true_label],
                    top1_predicted_label,
                    predict_score[i][top1_predicted_label]))
            cnt += 1
            if top1_predicted_label == true_label:
                acc_cnt += 1
            if top5_score[i]:
                acc5_cnt += 1      

        print("model %d - step %d of %d" % (modelid, step, number_of_steps_per_epoch))  

        #loss, acc = sess.run([cross_entropy, accuracy],feed_dict={images_placeholder: test_images,labels_placeholder: test_labels})
        #lab, pred, loss, acc = sess.run([true_class, predicted_class, cross_entropy, accuracy])
        #metrics_.append([loss, acc])
        #labels_.append(lab)
        #predicted_.append(pred)
    print("Test Accuracy={}".format(float(acc_cnt)/float(cnt)))
    print("Top-5 Accuracy={}".format(float(acc5_cnt)/float(cnt)))
    write_file.close()

    # metrics_ = np.array(metrics_)
    # np.save(savedir + "./loss_accuracy_model%d_data%d.npy" % modelid, metrics_)

    #labels_ = np.array(labels_)
    #np.save(savedir + "/truelabels_model%d_data%d.npy" % (modelid, dataid), labels_)

    #predicted_ = np.array(predicted_)
    #np.save(savedir + "/predictedlabels_model%d_data%d.npy" % (modelid, dataid), predicted_)


