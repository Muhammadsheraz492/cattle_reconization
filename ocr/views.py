from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import ImageUploadSerializer
from django.core.files.storage import FileSystemStorage
import cv2
import numpy as np
import imutils
import easyocr
import tensorflow as tf

# Load the pre-trained model
ocr_model = tf.keras.models.load_model('path/to/your/ocr_model.h5')

def handle_uploaded_image(image):
    img_array = np.frombuffer(image.read(), np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(gray, 50, 200)

    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    location = None

    for contour in contours:
        approx = cv2.approxPolyDP(contour, 10, True)
        if len(approx) == 4:
            location = approx
            break

    if location is not None:
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [location], 0, 255, -1)
        new_image = cv2.bitwise_and(img, img, mask=mask)

        (x, y) = np.where(mask == 255)
        (x1, y1) = (np.min(x), np.min(y))
        (x2, y2) = (np.max(x), np.max(y))
        cropped_image = gray[x1:x2 + 1, y1:y2 + 1]

        resized_image = cv2.resize(cropped_image, (28, 28))
        resized_image = resized_image.reshape((1, 28, 28, 1)).astype('float32') / 255

        prediction = ocr_model.predict(resized_image)
        predicted_label = np.argmax(prediction)

        reader = easyocr.Reader(['en'])
        result = reader.readtext(cropped_image)
        
        detected_texts = [{'text': text, 'confidence': prob} for (_, text, prob) in result]

        return predicted_label, detected_texts
    else:
        return None, []

class ImageUploadAPIView(APIView):
    def post(self, request, *args, **kwargs):
        serializer = ImageUploadSerializer(data=request.data)
        if serializer.is_valid():
            image = serializer.validated_data['image']
            predicted_label, detected_texts = handle_uploaded_image(image)
            return Response({
                'predicted_label': predicted_label,
                'detected_texts': detected_texts
            }, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
