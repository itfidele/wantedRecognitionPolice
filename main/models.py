from django.db import models

# Create your models here.

class Persons(models.Model):
    username=models.CharField(max_length=100)
    profile_image=models.ImageField(upload_to='users/photos/')
    
    def __str__(self):
        return self.username

    