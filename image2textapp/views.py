from django.shortcuts import render
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from image2text_model.class_prediction import *
from django.core.files.storage import FileSystemStorage
from image2text_model import class_prediction

obj = class_prediction()

# Create your views here.
@csrf_exempt
def index(request):
    return HttpResponse("Hello world!")

@csrf_exempt
def predict(request):
    if request.method=='POST':
        image = request.FILES['image']
        # print(image)
        fs = FileSystemStorage('static/')
        fs.save(image, image)
        output = obj.process(f'static/{image}')
        return HttpResponse(output)