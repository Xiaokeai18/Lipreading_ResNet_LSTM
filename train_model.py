import tensorflow as tf
from tensorflow.python.platform import tf_logging as logging
import pandas as pd
from model import *
#from tfpipeline import get_batch
import input_data

#--------------------------------------------------------------------------------------------------------------------#
# USER INPUT
# TRAIN OPTIONS
# data_info_dir = "/homes/mat10/Desktop/tfrecords_test"
#data_info_dir = "/data/mat10/ISO_Lipreading/data/LRW_TFRecords"
#train_data_info = pd.read_csv(data_info_dir + "/train_data_info.csv").sample(frac=1)
# val_data_info = pd.read_csv(data_info_dir + "/val_data_info.csv").sample(frac=1)
# words_step1 = ['BRITISH', 'ATTACKS', 'HAVING', 'BIGGEST', 'REPORT', 'FORCES',
#        'WANTED', 'HOURS', 'CONCERNS', 'INFORMATION']
# train_data_info = train_data_info[train_data_info['word'].isin(words_step1)]
train_options = {'batch_size': 8, 'num_classes': 500, 'num_epochs': 20,
                 'crop_size': 80, 'horizontal_flip': True, 'shuffle': True}

# MODEL RESTORE OPTIONS
restore = False
# specify the model directory
modeldir = "./saved_models_full"
model = "model_full_epoch19"

# SAVE OPTIONS
savedir = "./saved_models_full"

# START AT EPOCH
start_epoch = 0
#--------------------------------------------------------------------------------------------------------------------#
train_list_file = 'train.list'
num_train_videos = len(list(open(train_list_file,'r')))

print("Total number of train data: %d" % num_train_videos)
number_of_steps_per_epoch = num_train_videos // train_options['batch_size']
number_of_steps = train_options['num_epochs'] * number_of_steps_per_epoch

#train_paths = list(train_data_info['path'])

#--------------------------------------------------------------------------------------------------------------------#
# TRAINING
#train_videos, train_labels = get_batch(train_paths, train_options)
images_placeholder = tf.placeholder(tf.float32, shape=(train_options['batch_size'],24,train_options['crop_size']-20,train_options['crop_size'],3))
labels_placeholder = tf.placeholder(tf.int64, shape=(train_options['batch_size']))
prediction = frontend_3D(images_placeholder)
prediction = backend_resnet34(prediction)
prediction = blstm_2layer(prediction)
prediction = fully_connected_logits(prediction, train_options['num_classes'])

cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels_placeholder, logits=prediction))
tf.summary.scalar('cross_entropy', cross_entropy)

learning_rate = tf.train.exponential_decay(learning_rate=0.0001, global_step=0,
                                           decay_steps=number_of_steps_per_epoch//2,
                                           decay_rate=0.9, staircase=True)
tf.summary.scalar('learning_rate', learning_rate)

train_step = tf.train.MomentumOptimizer(learning_rate, momentum=0.9).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(prediction, axis=1), labels_placeholder)

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('accuracy', accuracy)

#logfile = open('saved_models_full/train_history_log.csv', 'w')
#logfile.write('epoch, step, train_accuracy \n')
config = tf.ConfigProto(allow_soft_placement = True)
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:

    # initialize the variables
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()

    # Add ops to save and restore all the variables.
    if restore:
        saver.restore(sess, modeldir + "/" + model)
        print("Model restored.")

    # print("saving model before training")
    # saver.save(sess=sess, save_path=savedir + "/model_full_epoch%d" % 0)
    # print("model saved")

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('./visual_logs/train', sess.graph)
    test_writer = tf.summary.FileWriter('./visual_logs/test', sess.graph)

    for epoch in range(start_epoch, start_epoch + train_options['num_epochs']):
        
        print("saving model for epoch %d - step %d" % (epoch, 0))
        saver.save(sess=sess, save_path=savedir + "/model_full_epoch%d" % epoch)
        print("model saved")
        
        for step in range(number_of_steps_per_epoch):
            train_images, train_labels, _, _, _ = input_data.read_clip_and_label(
                                filename='train.list',
                                batch_size=train_options['batch_size'],
                                num_frames_per_clip=24,
                                crop_size=train_options['crop_size'],
                                shuffle=True
                                )
            _, loss = sess.run([train_step, cross_entropy],feed_dict={images_placeholder: train_images,labels_placeholder: train_labels})
            #loss = sess.run( cross_entropy,feed_dict={images_placeholder: train_images,labels_placeholder: train_labels})
            

            if step%10 == 0:
                summary,train_acc = sess.run([merged,accuracy],feed_dict={images_placeholder: train_images,labels_placeholder: train_labels})
                train_writer.add_summary(summary, step+epoch*number_of_steps_per_epoch)
                print("epoch: %d of %d - step: %d of %d - loss: %.4f - train accuracy: %.4f"
                    % (epoch, train_options['num_epochs'], step, number_of_steps_per_epoch, loss, train_acc))

                #logfile.write('%d, %d, %.4f \n' % (epoch, step, train_acc))
                val_images, val_labels, _, _, _ = input_data.read_clip_and_label(
                                filename='test.list',
                                batch_size=train_options['batch_size'],
                                num_frames_per_clip=24,
                                crop_size=train_options['crop_size'],
                                shuffle=True
                                )
                summary,val_acc = sess.run([merged,accuracy],feed_dict={images_placeholder: val_images,labels_placeholder: val_labels})
                test_writer.add_summary(summary, step+epoch*number_of_steps_per_epoch)


    # stop our queue threads and properly close the session
    # coord.request_stop()
    # coord.join(threads)
    # sess.close()

#logfile.close()

#--------------------------------------------------------------------------------------------------------------------#



# #------------------------------------------------------------------------------------#
# # COMMENTED OUT
# prediction = frontend_3D(videos)
# prediction = backend_resnet34(prediction)
# prediction = blstm_2layer(prediction)
# prediction = fully_connected_logits(prediction, options['num_classes'])
#
# slim = tf.contrib.slim
#
# # sess = tf.Session()
# # sess.run(tf.initialize_all_variables())
# # tf.train.start_queue_runners(sess=sess)
# # d = sess.run([data])
#
# tf.losses.softmax_cross_entropy(labels, prediction)
# total_loss = slim.losses.get_total_loss()
#
# # Create some summaries to visualize the training process:
# # tf.scalar_summary('losses/Total Loss', total_loss)
#
# optimizer = tf.train.AdamOptimizer(learning_rate=0.005)
# train_op = slim.learning.create_train_op(total_loss,
#                                          optimizer,
#                                          summarize_gradients=True)
#
# logging.set_verbosity(1)
# slim.learning.train(train_op=train_op,
#                     number_of_steps=number_of_steps,
#                     logdir='ckpt/train',
#                     save_summaries_secs=60,
#                     save_interval_secs=600)
#------------------------------------------------------------------------------------#

# sess = tf.Session()
# sess.run(tf.initialize_all_variables())
# tf.train.start_queue_runners(sess=sess)
# pred = sess.run([prediction])
