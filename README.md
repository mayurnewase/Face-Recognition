# Face-Recognition
This project demonstrates power of one-shot learning for face recognition.
Here we use siamese model which is trained on 30 different people with 10 images per person.
It can identify them correctly,but now if we want it identify new person,model need only one image of that person for future recofnition.

For quickly running you can run it in flask server by
python scratch.py

open http://127.0.0.1:5000/ in browser.
Here you can test model on pretrained person images by entering name(s1,s2,..etc)and uploading any of his/her image from given dataset.
(currently model is only trained for first 30 person)

We can also quickly train model for other person by uploading his image with correct title and clicking on train.So in future model can correctly identify that person.
