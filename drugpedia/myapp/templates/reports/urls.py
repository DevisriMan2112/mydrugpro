# reports/urls.py
from django.urls import path
from views import download_report

urlpatterns = [
    path('download_report/', download_report, name='download_report'),
]
