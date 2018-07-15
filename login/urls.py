from django.conf.urls import url
from django.contrib.auth import views as auth_views
from login import views
app_name = 'login'


urlpatterns = [
    url(r'^$', view=views.index, name='index'),
    url(r'^login/$', auth_views.login, {'template_name': 'login/login.html'}, name='login'),
    url(r'^logout/$', auth_views.logout, name='logout'),
]
