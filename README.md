# Face-verification-using-One-shot-learning
Implementing one-shot learning using FaceNet

## Preface
Generally, deep learning algorithms are hungry for data and do not work well with a limited amount of data. This is where One-shot learning comes into the picture.

One-shot learning is a classification or categorization task in which one or very few data is used to classify many new data in the future. This technique is majorly used in facial verification to verify the identity of a certain someone.
In face verification task we compare the given face to the face that is already saved into the database. 

## But how a face is saved into the database?

We actually extract a feature vector of 128 number that defines your face, which is called Face Embeddings.<br>To make these face embedding we use a set of deep neural networks that takes the image vector as input and give a vector of 128 number vector as output. <br> Generally, Siamese Networks architecture is used for this purpose. 
In this project, I'm going to use a pre-trained model to make face embeddings for me named Facenet.

## Facenet:
FaceNet is a face recognition system developed in 2015 by researchers at Google that achieved then state-of-the-art results on a range of face recognition benchmark datasets.

The model is a deep convolutional neural network trained via a triplet loss function that encourages vectors for the same identity to become more similar (smaller distance), whereas vectors for different identities are expected to become less similar (larger distance). The focus on training a model to create embeddings directly (rather than extracting them from an intermediate layer of a model) was an important innovation in this work.

## Working:
* At first, to save a face into the database we run our input image through the facenet and get face embedding and save it in compressed form using a NumPy function savez_compressed().
* Now we have our registered user (the one saved in the database).
* Now next time, whenever you want to compare the new face with the saved one, then repeat the process up to getting face embedding, and now instead of saving it, we compared it with the saved one by calculating the vector distance between these two face embedding.
* We set a threshold value for the difference and if the difference is greater than the threshold then they are different persons else they are the same persons.
