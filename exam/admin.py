from django.contrib import admin
from exam.models import Student, Department, Course, Paper
# Register your models here.
admin.site.register(Student)
admin.site.register(Department)
admin.site.register(Course)
admin.site.register(Paper)