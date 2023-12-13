from django.urls import re_path
from .views import *

urlpatterns = [
    re_path(r'image_upload$', upload_query_image, name='image_upload'),
    re_path(r'success$', success, name='success'),
    re_path(r'images$', display_image, name='images'),
]