from django.urls import path
from . import views

urlpatterns = [
    path('', views.detect_faces, name='detect_faces'),
]