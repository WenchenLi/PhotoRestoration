import input_data

#use precropped 64x64 dataset.  write now some dimensions are hard coded.
dataset = input_data.read_data_sets("./data/celebACropped")

#for now, all we have is the training portion of the dataset.  You can access the training dataset with:
#dataset.train

#to get a batch of images:
#note that the batch size is variable, you can change it between individual batches if you want
imgs, masked_imgs = dataset.train.next_batch(64)

#the images and masked images make up the columns in the returned matrices
print "Size of 64 image batch:"
print imgs.shape
print masked_imgs.shape

print "Images are flattend and have range 0 - 1:"
print imgs[:,0]

#the number of epochs automatically updates as you request batches
print "Epochs completed after single batch:"
print dataset.train.epochs_completed
dataset.train.next_batch(100000)
dataset.train.next_batch(100000)
print "Epochs completed after two 100,000 image batches"
print dataset.train.epochs_completed
