# transfer-learning
I would like to thank vimal sir and preeti mam for their constant support in training different technologies and making us expertise in such technologies.
Here we used MobileNet model to achieve transfer learning in which the input shape is fixed according to models and top layer ie softmax layer is removed and all other layers are freeze.
MobileNet with operation on its layers
Then we added hidden layers through hit and trial and number of units in each layer and the output layer which will predict about the image and then combine the model using Model library in keras.
then using preprocessing library we used the image data generator function and augmented the data ie generate more images from existing one
we compile our model created and train our model which has some features of early stopping based on the loss with some stipulated number of allowance without any decrease in loss.
we can see from above pic that in first epoch the accuracy was 55 percent and till 20 epochs it increased to 98.8 percent .
then we took image of mr .colin and tried to ask our model about who he is it predicted 96 percent that he is mr. colin.
then we asked our model about another image and it predicted correct about mr. bush with 97 percent accuracy.
we can use any pre trained model for transfer learning such as inception ,vgg16,resnet50,mobilenet.they work same as above and are used helpful to person with low ram/cpu as it uses pre-trained weights.
this was my model with 98.8 percent accuracy with only total of 100 images of mr. colin and mr bush with accurate predictions.
once again thanking both of you vimal sir and preeti mam for effort you put in us.
parivansh deep singh
Rohit Bhatia
Shaurya khanna
