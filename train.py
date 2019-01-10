#-*- encoding:utf-8 -*-
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL.Image as Image
import model

#model save path, to restore the model
CHECK_POINT_DIR = 'D:/python-project/CatVsDogRecong/modelsave'

#log save path, to use tensorboard to see the loss and accuracy
LOG_DIR = 'D:/python-project/CatVsDogRecong/log'
def get_files(file_path):
	#two list to save train data and train label
	class_train = []
	label_train = []
	#train_class is the train image direcotry
	for train_class in os.listdir(file_path):
		#pic_name is the train image name
		for pic_name in os.listdir(file_path + train_class):
			class_train.append(file_path + train_class + '/' + pic_name)
			#train_class is 0,1,2,3,4....
			label_train.append(train_class)
	#merge trainimage and trainlabel to 2D array (2,n)
	temp = np.array([class_train, label_train])
	#transpose temp to (n,2)
	temp = temp.transpose()

	np.random.shuffle(temp)

	image_list = list(temp[:,0])
	label_list = list(temp[:,1])
	# class is 1 2 3 4 5 
	label_list = [int(i) for i in label_list]
	return image_list, label_list

def get_batches(image, label, resize_w, resize_h, batch_size, capacity):
	#tfansform imagelist to tf.string
	#transform label to tf.int64
	image = tf.cast(image, tf.string)
	label = tf.cast(label, tf.int64)
	queue = tf.train.slice_input_producer([image, label])
	label = queue[1]
	image_temp = tf.read_file(queue[0])
	image = tf.image.decode_jpeg(image_temp, channels = 3)
	#resize image 
	image = tf.image.resize_image_with_crop_or_pad(image, resize_w, resize_h)

	image = tf.image.per_image_standardization(image)

	image_batch, label_batch = tf.train.batch([image, label], batch_size = batch_size,
		num_threads = 64,
		capacity = capacity)
	images_batch = tf.cast(image_batch, tf.float32)
	labels_batch = tf.reshape(label_batch, [batch_size])
	return images_batch, labels_batch


train,train_label = get_files('D:/python-project/CatVsDogRecong/train_image/')
train_batch, train_label_batch = get_batches(train, train_label, 64, 64,64,1000)

train_logits = model.inference(train_batch,64,5)

train_loss = model.losses(train_logits, train_label_batch)

train_op = model.trainning(train_loss, 0.0001)

train_acc = model.evaluation(train_logits, train_label_batch)

summary_op = tf.summary.merge_all()

sess = tf.Session()
train_writer = tf.summary.FileWriter(LOG_DIR, sess.graph)
saver = tf.train.Saver()

sess.run(tf.global_variables_initializer())

coord = tf.train.Coordinator()


threads = tf.train.start_queue_runners(sess=sess, coord=coord)

try:
	ckpt = tf.train.get_checkpoint_state(CHECK_POINT_DIR) #断点续训
	if ckpt and ckpt.model_checkpoint_path:
		saver.restore(sess, ckpt.model_checkpoint_path)
	for step in np.arange(5000):
		if coord.should_stop():
			break
		_, tra_loss, tra_acc = sess.run([train_op, train_loss, train_acc])
		if step % 2== 0:
			print('Step %d, train loss=%.2f, train accuracy = %.2f%%' %(step, tra_loss, tra_acc))
			summary_str = sess.run(summary_op)
			train_writer.add_summary(summary_str, step)
		if (step + 1)%100==0:
			checkpoint_path = os.path.join(CHECK_POINT_DIR, 'model_ckpt')
			saver.save(sess, checkpoint_path, global_step=step)
except tf.errors.OutOfRangeError:
	print ('Done training')
finally:
	coord.request_stop()
coord.join(threads)