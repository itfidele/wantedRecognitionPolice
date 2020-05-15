import face_recognition
import numpy as np
from PIL import Image,ImageDraw
# Find faces in pictures
#image = face_recognition.load_image_file("/home/hacker/Desktop/pic/IMG_20190412_030129.jpg")
#face_locations = face_recognition.face_locations(image)
#print(face_locations)

# Find and manipulate facial features in pictures
# image = face_recognition.load_image_file("/home/hacker/Desktop/pic/IMG_20190412_030129.jpg")
# face_landmarks_list = face_recognition.face_landmarks(image)
# print(face_landmarks_list)

'''
known_image = face_recognition.load_image_file("/home/hacker/Desktop/pic/fidele.jpg")
unknown_image = face_recognition.load_image_file("/home/hacker/Desktop/pic/IMG_20190412_030129.jpg")
biden_encoding = face_recognition.face_encodings(known_image)[0]
unknown_encoding = face_recognition.face_encodings(unknown_image)[0]
results = face_recognition.compare_faces([biden_encoding], unknown_encoding)
print(results)
'''
picture_of_me = face_recognition.load_image_file("/home/hacker/Desktop/pic/IMG_20190412_030129.jpg")
try:
    my_face_encoding = face_recognition.face_encodings(picture_of_me)[0]
except IndexError:
    print ("there is no face in this Image")
# my_face_encoding now contains a universal 'encoding' of my facial features that can
#˓ → be compared to any other picture of a face!
#known_picture = face_recognition.load_image_file("/home/hacker/Desktop/pic/fidele.jpg")
#known_face_encoding = face_recognition.face_encodings(known_picture)[0]
# Now we can see the two face encodings are of the same person with `compare_faces`!
#pil_img=Image.fromarray(picture_of_me)
#draw=ImageDraw.Draw(pil_img)

#picture_of_me_location=face_recognition.face_locations(picture_of_me)
#picture_of_me_encoding=face_recognition.face_encodings(picture_of_me,picture_of_me_location)

known_face_names=[]
#known_face_names.append("ishimwe")
known_face_names.append("fidele")
known_face_names.append("parfait")

encodings=[]
encodings.append('fidele_face_encoding')
encodings.append('parfait_face_encoding')

images=[]
images.append("fidele")
images.append("parfait")

files=[]
files.append("/home/hacker/Desktop/pic/fidele.jpg")
files.append("/home/hacker/Desktop/pic/parfait.jpg")

for i in range(0,len(images)):
    images[i]=face_recognition.load_image_file(files[i])
    encodings[i]=face_recognition.face_encodings(images[i])[0]

known_face_encodings = encodings

# Load an image with an unknown face
unknown_image = face_recognition.load_image_file('/home/hacker/Desktop/pic/IMG_20190816_082047.jpg')

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
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding,tolerance=0.35)

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



'''
for (top, right, bottom, left), face_encoding in zip(picture_of_me_location, picture_of_me_encoding):
    # See if the face is a match for the known face(s)
    matches = face_recognition.compare_faces([known_face_encoding], face_encoding,tolerance=0.35)

    name = ""

    if matches[0]:
        name="fidele"
    else:
        name="other"
    #if True in matches:
        #name = "Fidele"

    # Or instead, use the known face with the smallest distance to the new face
    #face_distances = face_recognition.face_distance(known_face_encoding, face_encoding)
    #best_match_index = np.argmin(face_distances)
    #if matches[best_match_index]:
    #    name = known_face_names[best_match_index]


    # Draw a box around the face using the Pillow module
    draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

    # Draw a label with a name below the face
    text_width, text_height = draw.textsize(name)
    draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
    draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))
del draw
pil_img.show()
'''

#results = face_recognition.compare_faces([my_face_encoding], known_face_encoding)

'''
results = face_recognition.compare_faces([my_face_encoding], unknown_face_encoding)
if results[0] == True:
    print("It's a picture of me!")
else:
    print("It's not a picture of me!")
'''