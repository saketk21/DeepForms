import os
from django.http import HttpResponse
from django.shortcuts import render, redirect
from DeepForms import settings
from exam.models import Department, Course, Student, Paper
from mapping import mainFunction, splitString, errorHandling
from prediction.data_handling import decrypt
from exam import urls

def view_data(request):
    papers_data = Paper.objects.all()
    return render(request, template_name='exam/viewData.html', context={'papers_data': papers_data})


def dashboard(request):
    if request.method == 'POST':
        paper_code = request.POST.get('paperCode')
        prn = request.POST.get('prn')
        student_obj = Student.objects.get(prn=prn)
        course_code = request.POST.get('courseCode')
        course_obj = Course.objects.get(code=course_code)
        image_input = None
        if 'imageInput' in request.FILES:
            image_input = request.FILES['imageInput']
        print(request.FILES['imageInput'].name)
        paper_obj = Paper()
        paper_obj.student = student_obj
        paper_obj.course = course_obj
        paper_obj.image = image_input
        paper_obj.marks = ""
        if(Paper.objects.filter(student=student_obj, course=course_obj).exists()):
            return render(request, 'exam/dashboard.html', context={'error': 'The paper with given Course Code and PRN of student already exists.'})
        paper_obj.save()
        print("saved.")

        paper_obj = Paper.objects.get(student=student_obj, course=course_obj)
        image_name = paper_obj.image.url.split('/')
        # TODO: Solve platform dependency from image path
        image_path = os.path.join(settings.BASE_DIR, "media/paperPhotos/"+image_name[-1])
        print(image_path)
        marks = mainFunction(image_path)
        paper_obj.marks = marks
        paper_obj.save()
        lst, grandTotal = splitString(marks)
        tot_error = errorHandling(lst)
        zip_list = zip(tot_error, lst)
        for a,b in zip_list:
            print(a,b, sep=" $$ ")
        print('Finished Processing.')
        return render(request, 'exam/viewData.html', context={
        'prn': prn,
        'course_code': course_code,
        'paper_code': paper_code,
        'subQuestions': ['A','B','C','D','E','F', 'Total'],
        'lst': lst,
        'grandTotal': grandTotal,
        'tot_error': tot_error,
        })

    else:
        if request.user.is_authenticated:
            return render(request, template_name='exam/dashboard.html', context={})
        else:
            return HttpResponse('<h1 style="margin-top: 100px">Please Login to access Dashboard.</h1><a href="/">Go Back</a>')

