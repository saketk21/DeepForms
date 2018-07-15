from django.db import models


class Student(models.Model):
    prn = models.CharField(max_length=20, primary_key=True)


class Department(models.Model):
    dept_no = models.IntegerField(primary_key=True)
    name = models.CharField(max_length=40)


class Course(models.Model):
    code = models.CharField(max_length=10, primary_key=True)
    dept = models.ForeignKey(Department, on_delete=models.CASCADE)


class Paper(models.Model):
    code = models.IntegerField(primary_key=True)
    student = models.ForeignKey(Student, on_delete=models.CASCADE)
    course = models.ForeignKey(Course, on_delete=models.CASCADE)
    image = models.ImageField(upload_to='paperPhotos/')
    marks = models.CharField(max_length=1000, default="")
