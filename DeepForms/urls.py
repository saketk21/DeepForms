from django.conf.urls import url
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import include

from DeepForms import settings


urlpatterns = [
    url(r'^admin/', admin.site.urls),
    url(r'', include('login.urls')),
    url(r'^dashboard/', include('exam.urls')),
]
if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

