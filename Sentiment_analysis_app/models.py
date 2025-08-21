from django.db import models

# Create your models here.



class Review_DB(models.Model):
    User_query=models.CharField(max_length=100)
    classification=models.CharField(max_length=100)

    def __str__(self):
        return self.classification
