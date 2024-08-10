from django.contrib import admin

# Register your models here.
from django.contrib import admin
from .models import Feedback

# Register the Feedback model with the admin site
admin.site.register(Feedback)
