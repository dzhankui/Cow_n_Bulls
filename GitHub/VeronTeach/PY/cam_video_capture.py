import cv2

# Создаем объект для захвата видео с камеры (0 - индекс встроенной камеры)
cap = cv2.VideoCapture(0)

# Проверяем, успешно ли подключились к камере
if not cap.isOpened():
	print("Ошибка: камера не доступна")
	exit()

print("Для завершения нажмите 'q' в окне с изображением")

while True:
	# Захватываем кадр с камеры
	ret, frame = cap.read()
	
	# Если кадр не получен, выходим из цикла
	if not ret:
		print("Ошибка: не удалось получить кадр")
		break
	
	# Отображаем кадр в окне с именем "Webcam"
	cv2.imshow('Webcam', frame)
	
	# Прерываем цикл при нажатии клавиши 'q'
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# Освобождаем ресурсы камеры и закрываем окна
cap.release()
cv2.destroyAllWindows()