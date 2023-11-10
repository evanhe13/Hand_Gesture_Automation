# Hand_Gesture_Automation

This project allows you to control various actions on a Windows machine using hand gestures captured by an ESP32-CAM. 

## Materials

- [Arduino IDE](https://www.arduino.cc/en/software) or platform of your choice.
- [ESP32-CAM]([https://www.amazon.com/ESP32-CAM-MB-Aideepen-ESP32-CAM-Bluetooth-Arduino/dp/B0948ZFTQZ/ref=nav_signin?pd_rd_w=6Gozh&content-id=amzn1.sym.76a0b561-a7b4-41dc-9467-a85a2fa27c1c&pf_rd_p=76a0b561-a7b4-41dc-9467-a85a2fa27c1c&pf_rd_r=ZC7J1YECAFRRQ63EQSP6&pd_rd_wg=SEUPN&pd_rd_r=3a202dec-62d0-456d-b5b6-d29e17673c41&pd_rd_i=B0948ZFTQZ&psc=1]) module.
- [Wifi Antenna]([https://www.amazon.com/Diymall-Antenna-Antennas-Arduino-ESP-072pcs/dp/B00ZBJNO9O/ref=sr_1_2?crid=2X8JFRVZCAJ45&keywords=antenna+arduino&qid=1699657997&s=electronics&sprefix=antenna+arduino%2Celectronics%2C153&sr=1-2]) for ESP32-CAM.
- A local Wi-Fi network to connect your ESP32-CAM.

## Project Overview

The project consists of two parts:

### Part 1: Camera Stream with ESP32-CAM

The ESP32-CAM code in the [provided complementary code](#complementary-code-for-uploading-images-to-a-web-server) allows you to set up the ESP32-CAM module to capture photos and stream them to a web server. The web server runs locally and provides a live video feed of the camera. The server can be accessed through a web browser.

### Part 2: Hand Gesture Analysis

This part involves capturing screenshots from an online video feed and enhancing the selected screenshot using image processing. It then analyzes hand landmarks in the enhanced screenshot using the MediaPipe HandPose model. Different hand gestures, such as thumbs-up, thumbs-down, high-five, and a custom "Hook 'em" gesture, are detected. The detected gestures trigger specific actions, such as controlling system volume, showing/hiding the desktop, and opening a web link. 

## Author

This project was developed by [Evan He, Caleb Hererra, George Mathew, Jack Yeung].

## License

This project is open-source and available under the [MIT License](LICENSE).

## Acknowledgments

- This project is inspired by the work of various contributors and the [MediaPipe library](https://mediapipe.dev/).
- Special thanks to [Rui Santos](https://RandomNerdTutorials.com) for the ESP32-CAM complementary code.

## References

- [MediaPipe HandPose](https://mediapipe.dev/solutions/hands)
- [ESP32-CAM Guide by Random Nerd Tutorials](https://randomnerdtutorials.com/esp32-cam-take-photo-display-web-server/)
