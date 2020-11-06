import dlib
import face_recognition 
import os
import cv2


knownFacesDir = 'known_faces'
#unknownFacesDir = 'unknown_faces'
tolerance = 0.5
frameThickness = 3
fontThinkness = 2
model = 'hog'#hog

video = cv2.VideoCapture(0)


print("Loading Known Faces")

knownFaces = []
knownNames = []

for name in os.listdir(knownFacesDir):
	for fileName in os.listdir(f"{knownFacesDir}/{name}"):
		image = face_recognition.load_image_file(f"{knownFacesDir}/{name}/{fileName}")
		encoding = face_recognition.face_encodings(image)[0]
		knownFaces.append(encoding)
		knownNames.append(name)

print("Processing Unknown faces...")

while True:
	#print(filename)
	#image = face_recognition.load_image_file(f"{unknownFacesDir}/{filename}")
	ret, image = video.read()

	loactions = face_recognition.face_locations(image, model = model)
	encodings = face_recognition.face_encodings(image, loactions)
	#image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

	for face_encodings, face_locations in zip(encodings, loactions):
		results = face_recognition.compare_faces(knownFaces, face_encodings, tolerance)
		match = None
		if True in results:
			match = knownNames[results.index(True)]
			print(f"Match Found: {match}")
			top_left = (face_locations[3], face_locations[0])
			btm_left = (face_locations[1], face_locations[2])
			color = [255, 0 , 0]
			cv2.rectangle(image, top_left, btm_left, color, frameThickness)

			top_left = (face_locations[3], face_locations[2])
			btm_left = (face_locations[1], face_locations[2]+22)
			cv2.rectangle(image, top_left, btm_left, color, cv2.FILLED)
			cv2.putText(image, match, (face_locations[3]+10, face_locations[2]+15),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), fontThinkness)
	cv2.imshow(fileName, image)
	if cv2.waitKey(1) & 0xff == ord('q'):
		break

	#cv2.waitKey(0)
	#cv2.destroyWindow(fileName)



