# importing the cv2 library
import cv2

# loading the haar case algorithm file into alg variable
alg = "haarcascade_frontalface_default.xml"
# passing the algorithm to OpenCV
haar_cascade = cv2.CascadeClassifier(alg)
print("Cascade loaded OK:", not haar_cascade.empty())
# loading the image path into file_name variable - replace <INSERT YOUR IMAGE NAME HERE> with the path to your image
file_name = "brooklyn99.jpg"
# reading the image
img = cv2.imread(file_name, 0)
print("Image loaded:", img is not None)
if img is not None:
    print("Image shape:", img.shape)
# detecting the faces
faces = haar_cascade.detectMultiScale(
    img, scaleFactor=1.05, minNeighbors=5, minSize=(100, 100)
)
print("Faces found:", len(faces))

i = 0
# for each face detected
for x, y, w, h in faces:
    # crop the image to select only the face
    cropped_image = img[y : y + h, x : x + w]
    # loading the target image path into target_file_name variable  - replace kardashians with the path to your target image
    target_file_name = 'stored-faces/' + str(i) + '.jpg'
    cv2.imwrite(
        target_file_name,
        cropped_image,
    )
    i += 1