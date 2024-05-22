from django.contrib import admin
from .models import Record
admin.site.site_header = "laba"

admin.site.register(Record)
