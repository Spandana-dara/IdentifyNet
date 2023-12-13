from django.shortcuts import render, redirect
from django.http import HttpResponse, JsonResponse
from .forms import FaceImageForm
from .models import FaceImage
from .cluster_main import find_neighbours
from django.views.decorators.csrf import csrf_exempt


# Create your views here.
@csrf_exempt
def upload_query_image(request):
    if request.method == 'POST':
        form = FaceImageForm(request.POST, request.FILES)

        if form.is_valid():
            print(form.cleaned_data["name"])
            form.save()
            return redirect('images')
    else:
        form = FaceImageForm()
    return render(request, 'image_query.html', {'form': form})


def success(request):
    # return HttpResponse('Successfully uploaded!!')
    return JsonResponse(data={"hello": "by"})


def display_image(request):
    if request.method == 'GET':
        face_images = FaceImage.objects.last()
        path = '.' + face_images.face_img.url
        name = face_images.name
        content_form = find_neighbours(path, name)
        return render(request, 'display_images.html', {'face_images': content_form})

