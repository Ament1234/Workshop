import cv2
from deepface import DeepFace
# Load video from webcam
cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    if not ret:
        continue  # Skip if frame not captured

    try:
        # Analyze emotion
        results = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)

        if results and len(results) > 0:
            # Display emotion on frame
            emotion = results[0]['dominant_emotion']
            cv2.putText(img, f'Emotion: {emotion}', (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        else:
            cv2.putText(img, 'No face detected', (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    except Exception as e:
        print(f"Error analyzing emotion: {e}")
        cv2.putText(img, 'Analysis error', (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Emotion Recognition", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
