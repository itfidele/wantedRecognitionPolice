from django.shortcuts import render
from django.contrib.auth.decorators import login_required
# Create your views here.
from django.core.files.storage import FileSystemStorage
import face_recognition
import numpy as np
from PIL import Image,ImageDraw
from main.models import Persons

#@login_required()
def index(request):
    context={}
    if request.method== 'POST':
        fs=FileSystemStorage()
        file=request.FILES['file']
        filename=fs.save(file.name,file)
        uploaded_url=fs.url(filename)[1:]
        context['file']=uploaded_url
        detectWho(uploaded_url)
    return render(request,'index.html',context)


def detectWho(file_):

    known_face_names=[]
    encodings=[]
    images=[]
    files=[]

    pers=Persons.objects.all()
    for p in pers:
        encodings.append(p.username+"_face_encoding")
        images.append(p.username)
        known_face_names.append(p.username)
        files.append(p.profile_image)

    #files.append("/home/hacker/Desktop/pic/fidele.jpg")
    #files.append("/home/hacker/Desktop/pic/parfait.jpg")

    for i in range(0,len(images)):
        images[i]=face_recognition.load_image_file(files[i])
        encodings[i]=face_recognition.face_encodings(images[i])[0]

    known_face_encodings = encodings
    # Load an image with an unknown face
    unknown_image = face_recognition.load_image_file(file_)
    # Find all the faces and face encodings in the unknown image
    face_locations = face_recognition.face_locations(unknown_image)
    face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

    # Convert the image to a PIL-format image so that we can draw on top of it with the Pillow library
    # See http://pillow.readthedocs.io/ for more about PIL/Pillow
    pil_image = Image.fromarray(unknown_image)
    # Create a Pillow ImageDraw Draw instance to draw with
    draw = ImageDraw.Draw(pil_image)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding,tolerance=0.38)
        name = "Unknown"

        #If a match was found in known_face_encodings, just use the first one.
        #if True in matches:
        #    first_match_index = matches.index(True)
        #    name = known_face_names[first_match_index]

        # Or instead, use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        # Draw a box around the face using the Pillow module
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

        # Draw a label with a name below the face
        text_width, text_height = draw.textsize(name)
        draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
        draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))

        # Remove the drawing library from memory as per the Pillow docs
    del draw
    # Display the resulting image
    pil_image.show()