from django.urls import path
from . import views

urlpatterns = [

    path("predict/network", views.predict_network, name="predict/network"),
]
