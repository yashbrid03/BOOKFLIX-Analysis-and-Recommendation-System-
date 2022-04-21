from unicodedata import name
from django.urls import path
from . import views

urlpatterns = [
    path('', views.Home, name='Home'),
    path('index/', views.index, name='index'),
    path('recommend/', views.recommend, name='recommend'),
    path('login/', views.login, name='login'),
    path('postlogin/', views.postlogin, name='postlogin'),
    path('signup/', views.signup, name='signup'),
    path('logout/', views.logout, name='logout'),
    path('postsignup/', views.postsignup, name='postsignup'),
    path('shopping/', views.shopping, name='shopping'),
    path('postshopping/', views.postshopping, name='postshopping'),
    path('orders/', views.orders, name='orders')
]
