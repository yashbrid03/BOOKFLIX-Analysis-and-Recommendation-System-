from django.urls import path
from . import views

urlpatterns = [
    path('', views.Home, name='Home'),
    path('index/', views.index, name='index'),
    path('recommend/', views.recommend, name='recommend')
]
