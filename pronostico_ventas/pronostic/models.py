from django.db import models

# Create your models here.



from django.db import models

class User(models.Model):
    email = models.EmailField(max_length=30)
    password = models.CharField(max_length=50) 


class Persona(models.Model):
    nombre = models.CharField(max_length=100)
    apellido = models.CharField(max_length=100)



class CSVFile(models.Model):
    archivo = models.FileField(upload_to='archivos_csv/')
    fecha_subida = models.DateTimeField(auto_now_add=True)


