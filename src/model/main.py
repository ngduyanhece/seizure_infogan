import train
if __name__ == "__main__":
    cat_dim = (2,)
    noise_dim = (62,)
    batch_size = 64
    n_batch_per_epoch = 100
    nb_epoch = 1000
    #start to train
    train.train(cat_dim,noise_dim,batch_size,n_batch_per_epoch,nb_epoch)