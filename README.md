# DeepForms
DeepForms is a Deep Learning Web Application to recognize handwritten exam marks and populate data into RDBMS Table

### Usage
Being a Django Application, we can test-run it using the Django Development Web Server using ```python manage.py runserver```
The default URL will then be ```localhost:8000```

### Dependencies
1. Tensorflow - ```pip install tensorflow```
2. Keras - ```pip install keras```
3. Django - ```pip install django```

Other Dependencies exist, but they will be resolved during the ```pip install``` of the above.

### Test the Application
On running the application, upload any of the test images from the ```test_images``` folder.

### TODO:
If interested, you can submit a pull request regarding the following:
1. Asynchronous Processing of the Form/Paper.
2. Codebase is really dirty currently - Refactoring needed.
3. Improvements in Accuracy - Currently there are some wrong predictions of digits. Going to fix that.
