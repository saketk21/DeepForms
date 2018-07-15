from django.conf.urls import url
from exam import views

app_name = 'exam'

urlpatterns = [
    url(r'^$', view=views.dashboard, name='dashboard'),
    url(r'^viewData/', view=views.view_data, name='view_data')
]
